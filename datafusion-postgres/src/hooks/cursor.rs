use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use datafusion::common::{ParamValues, ToDFSchema};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser;
use datafusion::sql::sqlparser::ast::{CloseCursor, DeclareType, FetchDirection};
use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    catalog::streaming::StreamingTable,
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{stream::RecordBatchStreamAdapter, streaming::PartitionStream},
};
use futures::StreamExt;
use pgwire::api::ClientInfo;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::results::{Response, Tag};
use pgwire::api::stmt::StoredStatement;
use pgwire::api::store::{MemPortalStore, PortalStore};
use pgwire::error::{PgWireError, PgWireResult};
use tokio::sync::Mutex;

use super::{HookClient, QueryHook};
use crate::{arrow_pg::datatypes::df, handlers::RetainedPlan};

pub(crate) type DfStatement = crate::handlers::ParsedStatement;

/// Hook for processing cursor-related statements (DECLARE/FETCH/CLOSE)
#[derive(Debug)]
pub struct CursorStatementHook;

// The portal's retained scan owns the cursor; this index cannot keep it alive.
#[derive(Default)]
struct CursorRegistry(Mutex<HashMap<String, Weak<CursorSource>>>);

#[derive(Clone)]
struct CursorSource {
    schema: SchemaRef,
    rows: Arc<Mutex<CursorRows>>,
    limit: Option<usize>,
}

struct CursorRows {
    input: SendableRecordBatchStream,
    pending: Option<RecordBatch>,
}

impl std::fmt::Debug for CursorSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorSource")
            .field("schema", &self.schema)
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

impl CursorSource {
    fn dataframe(
        self: &Arc<Self>,
        ctx: &SessionContext,
    ) -> PgWireResult<datafusion::prelude::DataFrame> {
        let provider = StreamingTable::try_new(self.schema.clone(), vec![self.clone()])
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;
        ctx.read_table(Arc::new(provider))
            .map_err(|e| PgWireError::ApiError(Box::new(e)))
    }
}

impl PartitionStream for CursorSource {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _: Arc<TaskContext>) -> SendableRecordBatchStream {
        let input = self.rows.clone();
        let batches = futures::stream::try_unfold(Some((input, self.limit)), |state| async move {
            let Some((input, remaining)) = state else {
                return Ok(None);
            };
            if remaining == Some(0) {
                return Ok(None);
            }
            let mut rows = input.lock().await;
            let batch = match rows.pending.take() {
                Some(batch) => batch,
                None => match rows.input.next().await.transpose()? {
                    Some(batch) => batch,
                    None => return Ok(None),
                },
            };
            let count = remaining.unwrap_or(batch.num_rows()).min(batch.num_rows());
            if count < batch.num_rows() {
                rows.pending = Some(batch.slice(count, batch.num_rows() - count));
            }
            let remaining = remaining.map(|n| n - count);
            drop(rows);
            let next = (remaining != Some(0)).then_some((input, remaining));
            Ok(Some((batch.slice(0, count), next)))
        });
        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), batches))
    }
}

async fn cursor_source(client: &(dyn ClientInfo + Sync), name: &str) -> Option<Arc<CursorSource>> {
    let registry = client.session_extensions().get::<CursorRegistry>()?;
    let cursors = registry.0.lock().await;
    cursors.get(name).and_then(Weak::upgrade)
}

#[async_trait]
impl QueryHook for CursorStatementHook {
    async fn handle_simple_query(
        &self,
        statement: &sqlparser::ast::Statement,
        session_context: &SessionContext,
        client: &mut dyn HookClient,
    ) -> Option<PgWireResult<Response>> {
        handle_cursor(statement, session_context, client, &Format::UnifiedText).await
    }

    async fn handle_extended_parse_query(
        &self,
        statement: &sqlparser::ast::Statement,
        _session_context: &SessionContext,
        client: &(dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<LogicalPlan>> {
        let schema = match statement {
            sqlparser::ast::Statement::Fetch { name, .. } => {
                match cursor_source(client, &name.value).await {
                    Some(cursor) => match cursor.schema.clone().to_dfschema_ref() {
                        Ok(schema) => schema,
                        Err(e) => return Some(Err(PgWireError::ApiError(Box::new(e)))),
                    },
                    None => Arc::new(datafusion::common::DFSchema::empty()),
                }
            }
            sqlparser::ast::Statement::Declare { .. } | sqlparser::ast::Statement::Close { .. } => {
                Arc::new(datafusion::common::DFSchema::empty())
            }
            _ => return None,
        };
        Some(Ok(LogicalPlan::EmptyRelation(
            datafusion::logical_expr::EmptyRelation {
                produce_one_row: false,
                schema,
            },
        )))
    }

    async fn handle_extended_query(
        &self,
        statement: Option<&sqlparser::ast::Statement>,
        logical_plan: &LogicalPlan,
        params: &ParamValues,
        session_context: &SessionContext,
        client: &mut dyn HookClient,
    ) -> Option<PgWireResult<Response>> {
        self.handle_extended_query_formatted(
            statement,
            logical_plan,
            params,
            session_context,
            client,
            &Format::UnifiedBinary,
        )
        .await
    }

    async fn handle_extended_query_formatted(
        &self,
        statement: Option<&sqlparser::ast::Statement>,
        logical_plan: &LogicalPlan,
        _params: &ParamValues,
        session_context: &SessionContext,
        client: &mut dyn HookClient,
        format: &Format,
    ) -> Option<PgWireResult<Response>> {
        if let Some(sqlparser::ast::Statement::Fetch { name, .. }) = statement
            && let Some(cursor) = cursor_source(client, &name.value).await
            && logical_plan.schema().as_arrow() != cursor.schema.as_ref()
        {
            return Some(Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".into(),
                    "0A000".into(),
                    "cached plan must not change result type".into(),
                ),
            ))));
        }
        handle_cursor(statement?, session_context, client, format).await
    }
}

async fn handle_cursor(
    statement: &sqlparser::ast::Statement,
    context: &SessionContext,
    client: &mut dyn HookClient,
    format: &Format,
) -> Option<PgWireResult<Response>> {
    match statement {
        sqlparser::ast::Statement::Declare { stmts } => {
            Some(handle_declare(client, stmts, context).await)
        }
        sqlparser::ast::Statement::Fetch {
            name, direction, ..
        } => Some(handle_fetch(client, name, direction, context, format).await),
        sqlparser::ast::Statement::Close { cursor } => {
            Some(handle_close(client.portal_store(), cursor))
        }
        _ => None,
    }
}

async fn handle_declare(
    client: &mut dyn HookClient,
    stmts: &[datafusion::sql::sqlparser::ast::Declare],
    session_context: &SessionContext,
) -> PgWireResult<Response> {
    for declare in stmts {
        if declare.declare_type != Some(DeclareType::Cursor) {
            return Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".to_string(),
                    "42601".to_string(),
                    format!("unsupported DECLARE type: {:?}", declare.declare_type),
                ),
            )));
        }

        let cursor_name = match declare.names.first() {
            Some(name) => name.value.clone(),
            None => {
                return Err(PgWireError::UserError(Box::new(
                    pgwire::error::ErrorInfo::new(
                        "ERROR".to_string(),
                        "42601".to_string(),
                        "cursor name is required".to_string(),
                    ),
                )));
            }
        };

        let (for_query, ast) = match &declare.for_query {
            Some(q) => (q.to_string(), sqlparser::ast::Statement::Query(q.clone())),
            None => {
                return Err(PgWireError::UserError(Box::new(
                    pgwire::error::ErrorInfo::new(
                        "ERROR".to_string(),
                        "42601".to_string(),
                        "DECLARE CURSOR requires a FOR query".to_string(),
                    ),
                )));
            }
        };

        let df = session_context
            .sql(&for_query)
            .await
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

        let cursor = Arc::new(CursorSource {
            schema: Arc::new(df.schema().as_arrow().clone()),
            rows: Arc::new(Mutex::new(CursorRows {
                input: df
                    .execute_stream()
                    .await
                    .map_err(|e| PgWireError::ApiError(Box::new(e)))?,
                pending: None,
            })),
            limit: None,
        });
        let plan = cursor.dataframe(session_context)?.logical_plan().clone();
        let stored_stmt = Arc::new(StoredStatement::new(
            cursor_name.clone(),
            (for_query, Some((Some(ast), RetainedPlan::Held(plan)))),
            vec![],
        ));
        let portal = Portal::new_cursor(cursor_name.clone(), stored_stmt);
        client.portal_store().put_portal(Arc::new(portal));
        let registry = client
            .session_extensions()
            .get_or_insert_with(CursorRegistry::default);
        let mut cursors = registry.0.lock().await;
        cursors.retain(|_, cursor| cursor.strong_count() > 0);
        cursors.insert(cursor_name, Arc::downgrade(&cursor));
    }

    Ok(Response::Execution(Tag::new("DECLARE CURSOR")))
}

async fn handle_fetch(
    client: &mut dyn HookClient,
    name: &datafusion::sql::sqlparser::ast::Ident,
    direction: &FetchDirection,
    context: &SessionContext,
    format: &Format,
) -> PgWireResult<Response> {
    let cursor_name = &name.value;
    let store = client.portal_store();

    let max_rows = match direction {
        FetchDirection::Next | FetchDirection::Forward { limit: None } => Some(1),
        FetchDirection::Forward { limit: Some(v) } | FetchDirection::Count { limit: v } => {
            parse_value_as_usize(v)
        }
        FetchDirection::ForwardAll | FetchDirection::All => None,
        FetchDirection::Prior | FetchDirection::Backward { .. } | FetchDirection::BackwardAll => {
            return Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".to_string(),
                    "42000".to_string(),
                    "cursor can only scan forward".to_string(),
                ),
            )));
        }
        FetchDirection::First
        | FetchDirection::Last
        | FetchDirection::Absolute { .. }
        | FetchDirection::Relative { .. } => {
            return Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".to_string(),
                    "42000".to_string(),
                    "cursor can only scan forward".to_string(),
                ),
            )));
        }
    };

    // Like PostgreSQL's PerformPortalFetch: a missing portal is 34000
    // "cursor does not exist"; a portal bound from an empty query exists but
    // has no fetchable strategy and fails with XX000 "unsupported portal
    // strategy" (PortalRunFetch's default branch).
    let portal = match store.get_portal(cursor_name) {
        Some(entry) => match entry.value() {
            Some(portal) => Arc::clone(portal),
            None => {
                return Err(PgWireError::UserError(Box::new(
                    pgwire::error::ErrorInfo::new(
                        "ERROR".to_string(),
                        "XX000".to_string(),
                        "unsupported portal strategy".to_string(),
                    ),
                )));
            }
        },
        None => {
            return Err(PgWireError::UserError(Box::new(
                pgwire::error::ErrorInfo::new(
                    "ERROR".to_string(),
                    "34000".to_string(),
                    format!("cursor \"{cursor_name}\" does not exist"),
                ),
            )));
        }
    };

    if let Some(cursor) = cursor_source(client, cursor_name).await {
        if matches!(
            &*portal.state().lock().await,
            pgwire::api::portal::PortalExecutionState::Initial
        ) {
            let selected = Arc::new(CursorSource {
                limit: max_rows,
                ..cursor.as_ref().clone()
            });
            return Ok(Response::Query(
                df::encode_dataframe(selected.dataframe(context)?, format, None).await?,
            ));
        }
    }
    let fetch_result = portal.fetch(max_rows.unwrap_or(0)).await?;

    Ok(Response::Query(fetch_result.response))
}

fn handle_close(
    store: &MemPortalStore<DfStatement>,
    cursor: &CloseCursor,
) -> PgWireResult<Response> {
    match cursor {
        CloseCursor::All => {
            store.clear_portals();
        }
        CloseCursor::Specific { name } => {
            store.rm_portal(&name.value);
        }
    }
    Ok(Response::Execution(Tag::new("CLOSE CURSOR")))
}

fn parse_value_as_usize(value: &datafusion::sql::sqlparser::ast::Value) -> Option<usize> {
    match value {
        datafusion::sql::sqlparser::ast::Value::Number(s, _) => s.parse().ok(),
        _ => None,
    }
}
