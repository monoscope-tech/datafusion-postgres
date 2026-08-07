use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::ParamValues;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::*;
use datafusion::sql::parser::Statement;
use datafusion::sql::sqlparser;
use log::info;
use pgwire::api::auth::StartupHandler;
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::{ExtendedQueryHandler, SimpleQueryHandler};
use pgwire::api::results::{FieldInfo, Response, Tag};
use pgwire::api::stmt::QueryParser;
use pgwire::api::store::PortalStore;
use pgwire::api::{
    ClientInfo, ClientPortalStore, ConnectionManager, ErrorHandler, PgWireServerHandlers, Type,
};
use pgwire::error::{PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use pgwire::types::format::FormatOptions;

use crate::hooks::QueryHook;
use crate::hooks::cursor::CursorStatementHook;
use crate::hooks::set_show::SetShowHook;
use crate::hooks::transactions::TransactionStatementHook;
use crate::{client, planner};
use arrow_pg::datatypes::df;
use arrow_pg::datatypes::{arrow_schema_to_pg_fields, into_pg_type};
use datafusion_pg_catalog::sql::PostgresCompatibilityParser;

/// Rewrites Postgres command synonyms that DataFusion's SQL parser doesn't
/// recognize. Applied to every incoming SQL string before parsing — covers
/// both the simple-query and extended-query (parse) paths.
///
/// Currently handles:
/// - `ABORT [ WORK | TRANSACTION ]` → `ROLLBACK [ WORK | TRANSACTION ]`.
///   Postgres treats these as synonyms; Hasql's connection pool emits
///   `ABORT` defensively on session acquisition, which would otherwise
///   produce `sql parser error: Expected: an SQL statement, found: ABORT`
///   and poison the session.
///
/// Returns `Cow::Borrowed` on the no-rewrite fast path so the common case
/// pays only a short case-insensitive prefix check.
fn rewrite_postgres_synonyms(sql: &str) -> std::borrow::Cow<'_, str> {
    let stripped = sql.trim_start();
    if stripped.len() < 5 {
        return std::borrow::Cow::Borrowed(sql);
    }
    let (head, rest) = stripped.split_at(5);
    if !head.eq_ignore_ascii_case("ABORT") {
        return std::borrow::Cow::Borrowed(sql);
    }
    // Only treat as the command form when ABORT stands alone (not when it's
    // a prefix of an identifier like `aborted`).
    if !(rest.is_empty() || rest.starts_with(|c: char| c.is_whitespace() || c == ';')) {
        return std::borrow::Cow::Borrowed(sql);
    }
    std::borrow::Cow::Owned(format!("ROLLBACK{}", rest))
}

#[cfg(test)]
mod synonym_tests {
    use super::rewrite_postgres_synonyms as r;

    #[test]
    fn rewrites_abort_forms() {
        assert_eq!(r("ABORT"), "ROLLBACK");
        assert_eq!(r("ABORT;"), "ROLLBACK;");
        assert_eq!(r("  abort  "), "ROLLBACK  ");
        assert_eq!(r("Abort Work"), "ROLLBACK Work");
        assert_eq!(r("ABORT TRANSACTION;"), "ROLLBACK TRANSACTION;");
    }

    #[test]
    fn leaves_non_abort_alone() {
        assert_eq!(r("SELECT 1"), "SELECT 1");
        assert_eq!(r("BEGIN"), "BEGIN");
        assert_eq!(r("ROLLBACK"), "ROLLBACK");
        assert_eq!(r("SELECT aborted FROM t"), "SELECT aborted FROM t");
        assert_eq!(r("ABORTED"), "ABORTED");
    }
}

/// Simple startup handler that does no authentication
pub struct SimpleStartupHandler {
    connection_manager: Arc<ConnectionManager>,
}

#[async_trait::async_trait]
impl NoopStartupHandler for SimpleStartupHandler {
    fn connection_manager(&self) -> Option<Arc<ConnectionManager>> {
        Some(self.connection_manager.clone())
    }
}

pub struct HandlerFactory {
    pub session_service: Arc<DfSessionService>,
    cancel_handler: Arc<DefaultCancelHandler>,
    startup_handler: Arc<SimpleStartupHandler>,
}

impl HandlerFactory {
    pub fn new(session_context: Arc<SessionContext>) -> Self {
        let session_service = Arc::new(DfSessionService::new(session_context));
        let connection_manager = Arc::new(ConnectionManager::new());
        HandlerFactory {
            session_service,
            cancel_handler: Arc::new(DefaultCancelHandler::new(connection_manager.clone())),
            startup_handler: Arc::new(SimpleStartupHandler {
                connection_manager: connection_manager.clone(),
            }),
        }
    }

    pub fn new_with_hooks(
        session_context: Arc<SessionContext>,
        query_hooks: Vec<Arc<dyn QueryHook>>,
    ) -> Self {
        let session_service = Arc::new(DfSessionService::new_with_hooks(
            session_context,
            query_hooks,
        ));
        let connection_manager = Arc::new(ConnectionManager::new());
        HandlerFactory {
            session_service,
            cancel_handler: Arc::new(DefaultCancelHandler::new(connection_manager.clone())),
            startup_handler: Arc::new(SimpleStartupHandler {
                connection_manager: connection_manager.clone(),
            }),
        }
    }
}

impl PgWireServerHandlers for HandlerFactory {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        self.session_service.clone()
    }

    fn extended_query_handler(&self) -> Arc<impl ExtendedQueryHandler> {
        self.session_service.clone()
    }

    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        self.startup_handler.clone()
    }

    fn error_handler(&self) -> Arc<impl ErrorHandler> {
        Arc::new(LoggingErrorHandler)
    }

    fn cancel_handler(&self) -> Arc<impl CancelHandler> {
        self.cancel_handler.clone()
    }
}

struct LoggingErrorHandler;

impl ErrorHandler for LoggingErrorHandler {
    fn on_error<C>(&self, _client: &C, error: &mut PgWireError)
    where
        C: ClientInfo,
    {
        info!("Sending error: {error}")
    }
}

/// The pgwire handler backed by a datafusion `SessionContext`
pub struct DfSessionService {
    session_context: Arc<SessionContext>,
    parser: Arc<Parser>,
    query_hooks: Vec<Arc<dyn QueryHook>>,
}

impl DfSessionService {
    pub fn new(session_context: Arc<SessionContext>) -> DfSessionService {
        let hooks: Vec<Arc<dyn QueryHook>> = vec![
            Arc::new(CursorStatementHook),
            Arc::new(SetShowHook),
            Arc::new(TransactionStatementHook),
        ];
        Self::new_with_hooks(session_context, hooks)
    }

    pub fn new_with_hooks(
        session_context: Arc<SessionContext>,
        query_hooks: Vec<Arc<dyn QueryHook>>,
    ) -> DfSessionService {
        let parser = Arc::new(Parser {
            session_context: session_context.clone(),
            sql_parser: PostgresCompatibilityParser::new(),
            query_hooks: query_hooks.clone(),
        });
        DfSessionService {
            session_context,
            parser,
            query_hooks,
        }
    }
}

#[async_trait]
impl SimpleQueryHandler for DfSessionService {
    async fn do_query<C>(&self, client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo
            + ClientPortalStore
            + futures::Sink<PgWireBackendMessage>
            + Unpin
            + Send
            + Sync,
        C::PortalStore: PortalStore,
        C::Error: std::fmt::Debug,
        PgWireError: From<<C as futures::Sink<PgWireBackendMessage>>::Error>,
    {
        log::debug!("Received query: {query}");
        let rewritten = rewrite_postgres_synonyms(query);
        let query = rewritten.as_ref();
        let statements = self
            .parser
            .sql_parser
            .parse(query)
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

        // empty query
        if statements.is_empty() {
            return Ok(vec![Response::EmptyQuery]);
        }

        let mut results = vec![];
        'stmt: for statement in statements {
            // Call query hooks with the parsed statement
            for hook in &self.query_hooks {
                if let Some(result) = hook
                    .handle_simple_query(&statement, &self.session_context, client)
                    .await
                {
                    results.push(result?);
                    continue 'stmt;
                }
            }

            let df_result = {
                let query = statement.to_string();

                let timeout = client::get_statement_timeout(client);
                if let Some(timeout_duration) = timeout {
                    tokio::time::timeout(timeout_duration, self.session_context.sql(&query))
                        .await
                        .map_err(|_| {
                            PgWireError::UserError(Box::new(pgwire::error::ErrorInfo::new(
                                "ERROR".to_string(),
                                "57014".to_string(), // query_canceled error code
                                "canceling statement due to statement timeout".to_string(),
                            )))
                        })?
                } else {
                    self.session_context.sql(&query).await
                }
            };

            // Handle query execution errors and transaction state
            let df = match df_result {
                Ok(df) => df,
                Err(e) => {
                    return Err(PgWireError::ApiError(Box::new(e)));
                }
            };

            if let Some(resp) = dml_completion(&df).await? {
                results.push(resp);
            } else {
                let format_options =
                    Arc::new(FormatOptions::from_client_metadata(client.metadata()));
                results.push(Response::Query(
                    df::encode_dataframe(df, &Format::UnifiedText, Some(format_options)).await?,
                ));
            }
        }
        Ok(results)
    }
}

#[async_trait]
impl ExtendedQueryHandler for DfSessionService {
    type Statement = ParsedStatement;
    type QueryParser = Parser;

    fn query_parser(&self) -> Arc<Self::QueryParser> {
        self.parser.clone()
    }

    async fn do_query<C>(
        &self,
        client: &mut C,
        portal: &Portal<Self::Statement>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo
            + ClientPortalStore
            + futures::Sink<PgWireBackendMessage>
            + Unpin
            + Send
            + Sync,
        C::PortalStore: PortalStore,
        C::Error: std::fmt::Debug,
        PgWireError: From<<C as futures::Sink<PgWireBackendMessage>>::Error>,
    {
        let query = &portal.statement.statement.0;
        log::debug!("Received execute extended query: {query}");
        // Check query hooks first
        if !self.query_hooks.is_empty()
            && let (_, Some((statement, plan))) = &portal.statement.statement
        {
            // TODO: in the case where query hooks all return None, we do the param handling again later.
            let param_types = planner::get_inferred_parameter_types(plan)
                .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

            let param_values: ParamValues =
                df::deserialize_parameters(portal, &ordered_param_types(&param_types))?;

            for hook in &self.query_hooks {
                if let Some(result) = hook
                    .handle_extended_query(
                        statement.as_ref(),
                        plan,
                        &param_values,
                        &self.session_context,
                        client,
                    )
                    .await
                {
                    return result;
                }
            }
        }

        if let (_, Some((_statement, plan))) = &portal.statement.statement {
            let param_types = planner::get_inferred_parameter_types(plan)
                .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

            let mut param_values =
                df::deserialize_parameters(portal, &ordered_param_types(&param_types))?;

            // Let hooks append fresh values for placeholders they injected at
            // parse time beyond the client's binds (TimeFusion: a fresh now()
            // per execute for shape-cached now()+$N queries). Appended in $N
            // order after the client's params; matched by placeholder id, so
            // surplus is harmless.
            let extra: Vec<_> = self
                .query_hooks
                .iter()
                .flat_map(|h| h.extra_execute_params(_statement.as_ref()))
                .collect();
            if !extra.is_empty() && let ParamValues::List(list) = &mut param_values {
                list.extend(extra.into_iter().map(Into::into));
            }

            let plan = plan
                .clone()
                .replace_params_with_values(&param_values)
                .map_err(|e| PgWireError::ApiError(Box::new(e)))?;
            // Skip per-query `state.optimize()` ONLY when some hook
            // pre-optimized the plan at parse time (TimeFusion's PlanCacheHook
            // does this). Plans that fell through the bypass paths
            // (statement_to_plan in `do_parse_query`) are still optimized
            // here. Measured: skipping the redundant optimize on the cached
            // path dropped pgwire end-to-end p95 from 131ms → 8ms (~16×).
            let canonical_sql = &portal.statement.statement.0;
            let pre_optimized = self.query_hooks.iter().any(|h| h.was_pre_optimized(canonical_sql));
            let optimised = if pre_optimized {
                plan
            } else {
                self.session_context
                    .state()
                    .optimize(&plan)
                    .map_err(|e| PgWireError::ApiError(Box::new(e)))?
            };

            let dataframe = {
                let timeout = client::get_statement_timeout(client);
                if let Some(timeout_duration) = timeout {
                    tokio::time::timeout(
                        timeout_duration,
                        self.session_context.execute_logical_plan(optimised),
                    )
                    .await
                    .map_err(|_| {
                        PgWireError::UserError(Box::new(pgwire::error::ErrorInfo::new(
                            "ERROR".to_string(),
                            "57014".to_string(), // query_canceled error code
                            "canceling statement due to statement timeout".to_string(),
                        )))
                    })?
                    .map_err(|e| PgWireError::ApiError(Box::new(e)))?
                } else {
                    self.session_context
                        .execute_logical_plan(optimised)
                        .await
                        .map_err(|e| PgWireError::ApiError(Box::new(e)))?
                }
            };

            if let Some(resp) = dml_completion(&dataframe).await? {
                Ok(resp)
            } else {
                let format_options =
                    Arc::new(FormatOptions::from_client_metadata(client.metadata()));
                Ok(Response::Query(
                    df::encode_dataframe(
                        dataframe,
                        &portal.result_column_format,
                        Some(format_options),
                    )
                    .await?,
                ))
            }
        } else {
            Ok(Response::EmptyQuery)
        }
    }
}

/// If `df` runs a DML/COPY plan, execute it and return a `CommandComplete`
/// response with the right tag; otherwise return `None` so the caller falls
/// back to the regular `Response::Query` path. Driving this off
/// `LogicalPlan` (not the parsed AST) keeps the simple- and extended-query
/// paths consistent with what DataFusion actually runs — statement-level
/// rewrites can leave the AST in a non-Insert variant for what's really a write.
async fn dml_completion(df: &DataFrame) -> PgWireResult<Option<Response>> {
    use datafusion::arrow::array::UInt64Array;
    use datafusion::logical_expr::dml::WriteOp;
    let tag = match df.logical_plan() {
        LogicalPlan::Dml(d) => match d.op {
            WriteOp::Insert(_) => Tag::new("INSERT").with_oid(0),
            WriteOp::Update => Tag::new("UPDATE"),
            WriteOp::Delete => Tag::new("DELETE"),
            WriteOp::Ctas => Tag::new("SELECT"),
            WriteOp::Truncate => Tag::new("TRUNCATE"),
        },
        LogicalPlan::Copy(_) => Tag::new("COPY"),
        _ => return Ok(None),
    };
    let batches = df
        .clone()
        .collect()
        .await
        .map_err(|e| PgWireError::ApiError(Box::new(e)))?;
    let rows = batches
        .first()
        .and_then(|b| b.column_by_name("count"))
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .map_or(0, |a| a.value(0) as usize);
    Ok(Some(Response::Execution(tag.with_rows(rows))))
}

/// A prepared statement as the portal store holds it, for the life of the
/// statement: canonical SQL, plus the planned form when the text parsed to one.
///
/// The AST is `Option` because a prepared statement PINS everything in here.
/// A bulk `INSERT ... VALUES (...), (...), …` parses to an AST proportional to
/// the payload, and on TimeFusion's ingest path that AST was the single largest
/// resident object in the process — 35-42% of live heap across mid-run jemalloc
/// dumps (2026-08-07), growing ~1 GB/min because every distinct batch size is a
/// distinct prepared statement the client never closes. Nothing reads it after
/// Parse: execution runs off the `LogicalPlan`, and the hooks that dispatch on
/// statement kind only ever match control statements. See [`retains_ast`].
pub type ParsedStatement = (
    String,
    Option<(Option<sqlparser::ast::Statement>, LogicalPlan)>,
);

/// Whether a statement's AST is worth pinning for the life of the prepared
/// statement. Bulk data statements are the ones that get huge and the ones no
/// execute-time consumer needs: `handle_extended_query` implementations match
/// on SET / SHOW / DECLARE / transaction control, and `extra_execute_params`
/// only injects into `Statement::Query`. Everything else is small enough that
/// keeping it costs nothing.
fn retains_ast(statement: &sqlparser::ast::Statement) -> bool {
    !matches!(
        statement,
        sqlparser::ast::Statement::Insert(_)
            | sqlparser::ast::Statement::Update { .. }
            | sqlparser::ast::Statement::Delete(_)
            | sqlparser::ast::Statement::Copy { .. }
    )
}

fn retained_ast(statement: sqlparser::ast::Statement) -> Option<sqlparser::ast::Statement> {
    retains_ast(&statement).then_some(statement)
}

pub struct Parser {
    session_context: Arc<SessionContext>,
    sql_parser: PostgresCompatibilityParser,
    query_hooks: Vec<Arc<dyn QueryHook>>,
}

#[async_trait]
impl QueryParser for Parser {
    type Statement = ParsedStatement;

    async fn parse_sql<C>(
        &self,
        client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<Self::Statement>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        log::debug!("Received parse extended query: {sql}");
        let rewritten = rewrite_postgres_synonyms(sql);
        let sql = rewritten.as_ref();
        let mut statements = self
            .sql_parser
            .parse(sql)
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;
        if statements.is_empty() {
            return Ok((sql.to_string(), None));
        }

        let statement = statements.remove(0);
        let query = statement.to_string();

        let context = &self.session_context;
        let state = context.state();

        for hook in &self.query_hooks {
            if let Some(logical_plan) = hook
                .handle_extended_parse_query(&statement, context, client)
                .await
            {
                return Ok((query, Some((retained_ast(statement), logical_plan?))));
            }
        }

        let logical_plan = state
            .statement_to_plan(Statement::Statement(Box::new(statement.clone())))
            .await
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;
        Ok((query, Some((retained_ast(statement), logical_plan))))
    }

    fn get_parameter_types(&self, stmt: &Self::Statement) -> PgWireResult<Vec<Type>> {
        if let (_, Some((statement, plan))) = stmt {
            let params = planner::get_inferred_parameter_types(plan)
                .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

            let mut param_types = Vec::with_capacity(params.len());
            for param_type in ordered_param_types(&params).iter() {
                if let Some(datatype) = param_type {
                    let pgtype = into_pg_type(datatype)?;
                    param_types.push(pgtype);
                } else {
                    param_types.push(Type::UNKNOWN);
                }
            }

            // Hide hook-injected trailing placeholders (e.g. now() the plan cache
            // parameterized above the client's binds) so the client's
            // ParameterDescription still reports only its own params. They are
            // the highest-numbered ($N) placeholders, so they sort last.
            let injected: usize = self.query_hooks.iter().map(|h| h.injected_param_count(statement.as_ref())).sum();
            param_types.truncate(param_types.len().saturating_sub(injected));

            Ok(param_types)
        } else {
            Ok(vec![])
        }
    }

    fn get_result_schema(
        &self,
        stmt: &Self::Statement,
        column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        let Some((_, plan)) = stmt.1.as_ref() else {
            return Ok(vec![]);
        };
        let schema = plan.schema();
        let fields = schema.fields();
        // DataFusion emits `[count: UInt64]` for every DML/COPY plan — see
        // `make_count_schema` in datafusion/expr/src/logical_plan/dml.rs.
        // pgwire's contract for these without RETURNING is NoData; strict
        // clients reject a TuplesOk/NoData mismatch at Describe time. Match
        // on the exact shape so RETURNING (wider schema) and any future
        // upstream rename (e.g. `rows_affected`) fall through to the real
        // result path — at which point this guard needs to be updated.
        // (More precise than upstream #329's blanket Dml/Ddl→NoData, which
        // would drop RETURNING columns.)
        if matches!(plan, LogicalPlan::Dml(_) | LogicalPlan::Copy(_))
            && fields.len() == 1
            && fields[0].name() == "count"
            && fields[0].data_type() == &DataType::UInt64
        {
            return Ok(vec![]);
        }
        arrow_schema_to_pg_fields(
            schema.as_arrow(),
            column_format.unwrap_or(&Format::UnifiedBinary),
            None,
        )
    }
}

fn ordered_param_types(types: &HashMap<String, Option<DataType>>) -> Vec<Option<&DataType>> {
    // Datafusion stores the parameters as a map.  In our case, the keys will be
    // `$1`, `$2` etc.  The values will be the parameter types.
    //
    // PATCH (timefusion): original implementation sorted lexicographically
    // (`a.0.cmp(b.0)`), which puts `$10` before `$2` and breaks every
    // INSERT/SELECT with more than 9 placeholders — the ParameterDescription
    // returned to the client has the wrong positional order, so e.g. a uuid
    // gets typed as TIMESTAMPTZ. Sort by the numeric suffix instead.
    let mut entries: Vec<_> = types.iter().collect();
    entries.sort_by_key(|(k, _)| k.trim_start_matches('$').parse::<u32>().unwrap_or(u32::MAX));
    entries.into_iter().map(|pt| pt.1.as_ref()).collect()
}

#[cfg(test)]
mod tests {
    use datafusion::prelude::SessionContext;

    use super::*;
    use crate::testing::MockClient;

    use crate::hooks::HookClient;

    struct TestHook;

    #[async_trait]
    impl QueryHook for TestHook {
        async fn handle_simple_query(
            &self,
            statement: &sqlparser::ast::Statement,
            _ctx: &SessionContext,
            _client: &mut dyn HookClient,
        ) -> Option<PgWireResult<Response>> {
            if statement.to_string().contains("magic") {
                Some(Ok(Response::EmptyQuery))
            } else {
                None
            }
        }

        async fn handle_extended_parse_query(
            &self,
            _statement: &sqlparser::ast::Statement,
            _session_context: &SessionContext,
            _client: &(dyn ClientInfo + Send + Sync),
        ) -> Option<PgWireResult<LogicalPlan>> {
            None
        }

        async fn handle_extended_query(
            &self,
            _statement: Option<&sqlparser::ast::Statement>,
            _logical_plan: &LogicalPlan,
            _params: &ParamValues,
            _session_context: &SessionContext,
            _client: &mut dyn HookClient,
        ) -> Option<PgWireResult<Response>> {
            None
        }
    }

    /// A prepared statement pins whatever `parse_sql` returns. Bulk data
    /// statements must not pin their AST (2026-08-07: 35-42% of TimeFusion's
    /// live heap), and everything the execute-time hooks dispatch on must.
    #[test]
    fn bulk_data_statements_do_not_pin_their_ast() {
        let parse = |sql: &str| {
            PostgresCompatibilityParser::new()
                .parse(sql)
                .unwrap()
                .remove(0)
        };
        for sql in [
            "INSERT INTO t VALUES (1), (2)",
            "UPDATE t SET a = 1",
            "DELETE FROM t",
        ] {
            assert!(retained_ast(parse(sql)).is_none(), "must not pin: {sql}");
        }
        for sql in [
            "SET x = 1",
            "SHOW ALL",
            "BEGIN",
            "COMMIT",
            "DECLARE c CURSOR FOR SELECT 1",
            "SELECT 1",
        ] {
            assert!(retained_ast(parse(sql)).is_some(), "must pin: {sql}");
        }
    }

    #[tokio::test]
    async fn test_query_hooks() {
        let hook = TestHook;
        let ctx = SessionContext::new();
        let mut client = MockClient::new();

        // Parse a statement that contains "magic"
        let parser = PostgresCompatibilityParser::new();
        let statements = parser.parse("SELECT magic").unwrap();
        let stmt = &statements[0];

        // Hook should intercept
        let result = hook.handle_simple_query(stmt, &ctx, &mut client).await;
        assert!(result.is_some());

        // Parse a normal statement
        let statements = parser.parse("SELECT 1").unwrap();
        let stmt = &statements[0];

        // Hook should not intercept
        let result = hook.handle_simple_query(stmt, &ctx, &mut client).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_multiple_statements_with_hook_continue() {
        // Bug #227: when a hook returned a result, the code used `break 'stmt`
        // which would exit the entire statement loop, preventing subsequent statements
        // from being processed.
        let session_context = Arc::new(SessionContext::new());

        let hooks: Vec<Arc<dyn QueryHook>> = vec![Arc::new(TestHook)];
        let service = DfSessionService::new_with_hooks(session_context, hooks);

        let mut client = MockClient::new();

        // Mix of queries with hooks and those without
        let query = "SELECT magic; SELECT 1; SELECT magic; SELECT 1";

        let results =
            <DfSessionService as SimpleQueryHandler>::do_query(&service, &mut client, query)
                .await
                .unwrap();

        assert_eq!(results.len(), 4, "Expected 4 responses");

        assert!(matches!(results[0], Response::EmptyQuery));
        assert!(matches!(results[1], Response::Query(_)));
        assert!(matches!(results[2], Response::EmptyQuery));
        assert!(matches!(results[3], Response::Query(_)));
    }

    #[tokio::test]
    async fn test_set_sends_parameter_status_via_sink() {
        use pgwire::messages::PgWireBackendMessage;

        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        let test_cases = vec![
            ("SET datestyle = 'ISO, MDY'", "DateStyle", "ISO, MDY"),
            (
                "SET intervalstyle = 'postgres'",
                "IntervalStyle",
                "postgres",
            ),
            ("SET bytea_output = 'hex'", "bytea_output", "hex"),
            (
                "SET application_name = 'myapp'",
                "application_name",
                "myapp",
            ),
            ("SET search_path = 'public'", "search_path", "public"),
            ("SET extra_float_digits = '2'", "extra_float_digits", "2"),
            (
                "SET TIME ZONE 'America/New_York'",
                "TimeZone",
                "America/New_York",
            ),
        ];

        for (sql, expected_key, expected_value) in test_cases {
            client.sent_messages.clear();

            let responses =
                <DfSessionService as SimpleQueryHandler>::do_query(&service, &mut client, sql)
                    .await
                    .unwrap();

            assert!(
                matches!(responses[0], Response::Execution(_)),
                "Expected SET tag for {sql}"
            );

            let ps_msgs: Vec<_> = client
                .sent_messages()
                .iter()
                .filter_map(|m| match m {
                    PgWireBackendMessage::ParameterStatus(ps) => Some(ps),
                    _ => None,
                })
                .collect();

            assert_eq!(ps_msgs.len(), 1, "Expected 1 ParameterStatus for {sql}");
            assert_eq!(ps_msgs[0].name, expected_key, "Wrong key for {sql}");
            assert_eq!(ps_msgs[0].value, expected_value, "Wrong value for {sql}");
        }
    }

    #[tokio::test]
    async fn test_set_statement_timeout_no_parameter_status() {
        use pgwire::messages::PgWireBackendMessage;

        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "SET statement_timeout TO '5000ms'",
        )
        .await
        .unwrap();

        let has_ps = client
            .sent_messages()
            .iter()
            .any(|m| matches!(m, PgWireBackendMessage::ParameterStatus(_)));

        assert!(!has_ps, "statement_timeout should not send ParameterStatus");
    }

    /// `Describe Statement` for INSERT/UPDATE/DELETE without RETURNING must
    /// return an empty result schema so pgwire emits `NoData`. Strict clients
    /// (Hasql, pgjdbc, Npgsql, psycopg3, sqlx) treat a `RowDescription` here
    /// as a `TuplesOk` protocol error and drop the write. SELECT is the
    /// fallthrough positive control.
    #[tokio::test]
    async fn get_result_schema_returns_no_data_for_dml() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "CREATE TABLE t (id INT, name TEXT)",
        )
        .await
        .unwrap();

        let parser = <DfSessionService as ExtendedQueryHandler>::query_parser(&service);
        let cases: &[(&str, bool)] = &[
            ("INSERT INTO t VALUES (1, 'a')", true),
            ("UPDATE t SET name = 'x' WHERE id = 1", true),
            ("DELETE FROM t WHERE id = 1", true),
            // COPY: not tested — `state.statement_to_plan` rejects COPY as
            // unsupported today, so `LogicalPlan::Copy` is unreachable via
            // the prepared-statement path. The Copy arm in the guard is
            // defensive for if upstream ever enables it.
            ("SELECT id, name FROM t", false),
            // Over-match guard: a SELECT that happens to produce a single
            // UInt64 `count` column must NOT be suppressed — only DML/COPY
            // plans of that shape may. If the guard ever drops the
            // `LogicalPlan::Dml | Copy` check, this case fails loudly.
            ("SELECT COUNT(*) AS count FROM t", false),
        ];
        for (sql, expect_empty) in cases {
            let stmt = parser.parse_sql(&client, sql, &[]).await.unwrap();
            let fields = parser.get_result_schema(&stmt, None).unwrap();
            assert_eq!(
                fields.is_empty(),
                *expect_empty,
                "{sql}: expected empty={expect_empty}, got {fields:?}"
            );
        }
    }

    fn assert_execution_tag(response: &Response, expected: &str) {
        match response {
            Response::Execution(tag) => {
                let cc = pgwire::messages::response::CommandComplete::from(tag.clone());
                assert_eq!(cc.tag, expected, "Unexpected execution tag");
            }
            other => panic!("Expected Execution response, got: {other:?}"),
        }
    }

    async fn assert_query_response_empty(response: &mut Response) {
        use futures::StreamExt;

        let Response::Query(qr) = response else {
            panic!("Expected Query response, got: {response:?}");
        };

        let mut count = 0;
        while qr.data_rows().next().await.is_some() {
            count += 1;
        }
        assert_eq!(count, 0, "Expected no rows from exhausted cursor");
    }

    #[tokio::test]
    async fn test_declare_fetch_close_cursor() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE test_cursor CURSOR FOR SELECT 1 AS col",
        )
        .await
        .unwrap();

        assert_eq!(responses.len(), 1);
        assert_execution_tag(&responses[0], "DECLARE CURSOR");

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH NEXT FROM test_cursor",
        )
        .await
        .unwrap();

        assert_eq!(responses.len(), 1);
        assert!(
            matches!(&responses[0], Response::Query(_)),
            "Expected Query response for FETCH"
        );

        let mut responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH NEXT FROM test_cursor",
        )
        .await
        .unwrap();

        assert_eq!(responses.len(), 1);
        assert_query_response_empty(&mut responses[0]).await;

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "CLOSE test_cursor",
        )
        .await
        .unwrap();

        assert_eq!(responses.len(), 1);
        assert_execution_tag(&responses[0], "CLOSE CURSOR");
    }

    #[tokio::test]
    async fn test_fetch_nonexistent_cursor() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        let result = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH NEXT FROM nonexistent",
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_close_all_portals() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE c1 CURSOR FOR SELECT 1",
        )
        .await
        .unwrap();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE c2 CURSOR FOR SELECT 2",
        )
        .await
        .unwrap();

        let responses =
            <DfSessionService as SimpleQueryHandler>::do_query(&service, &mut client, "CLOSE ALL")
                .await
                .unwrap();

        assert!(matches!(&responses[0], Response::Execution(_)),);

        let result = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH NEXT FROM c1",
        )
        .await;
        assert!(result.is_err(), "c1 should be closed");
    }

    #[tokio::test]
    async fn test_fetch_forward_n() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "CREATE TABLE nums AS SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5",
        )
        .await
        .unwrap();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE mycur CURSOR FOR SELECT n FROM nums ORDER BY n",
        )
        .await
        .unwrap();

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH FORWARD 3 FROM mycur",
        )
        .await
        .unwrap();

        assert!(
            matches!(&responses[0], Response::Query(_)),
            "Expected Query response for FORWARD 3"
        );

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH FORWARD ALL FROM mycur",
        )
        .await
        .unwrap();

        let resp_desc = match &responses[0] {
            Response::Query(_) => "Query".to_string(),
            Response::Execution(tag) => {
                let cc = pgwire::messages::response::CommandComplete::from(tag.clone());
                format!("Execution({})", cc.tag)
            }
            other => format!("{:?}", other),
        };
        assert!(
            matches!(&responses[0], Response::Query(_)),
            "Expected Query response for remaining rows, got: {resp_desc}"
        );

        let mut responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH NEXT FROM mycur",
        )
        .await
        .unwrap();

        assert_query_response_empty(&mut responses[0]).await;
    }

    #[tokio::test]
    async fn test_scroll_cursor_error() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE mycur CURSOR FOR SELECT 1",
        )
        .await
        .unwrap();

        let result = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH PRIOR FROM mycur",
        )
        .await;

        assert!(result.is_err(), "PRIOR should fail on forward-only cursor");
    }

    #[tokio::test]
    async fn test_move_cursor() {
        let service = crate::testing::setup_handlers();
        let mut client = MockClient::new();

        <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "DECLARE mycur CURSOR FOR SELECT generate_series(1, 5) AS n",
        )
        .await
        .unwrap();

        let responses = <DfSessionService as SimpleQueryHandler>::do_query(
            &service,
            &mut client,
            "FETCH FORWARD 3 FROM mycur",
        )
        .await
        .unwrap();

        assert!(matches!(&responses[0], Response::Query(_)));
    }
}
