pub mod cursor;
pub mod permissions;
pub mod set_show;
pub mod transactions;

use async_trait::async_trait;

use datafusion::common::{ParamValues, ScalarValue};
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::Statement;
use futures::Sink;
use pgwire::api::ClientInfo;
use pgwire::api::ClientPortalStore;
use pgwire::api::results::Response;
use pgwire::api::store::{MemPortalStore, PortalStore};
use pgwire::error::{PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;

use crate::hooks::cursor::DfStatement;

#[async_trait]
pub trait HookClient: ClientInfo + Send + Sync {
    fn portal_store(&self) -> &MemPortalStore<DfStatement>;

    async fn send_message(&mut self, item: PgWireBackendMessage) -> PgWireResult<()>;
}

#[async_trait]
impl<S> HookClient for S
where
    S: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Send + Sync + Unpin,
    PgWireError: From<<S as Sink<PgWireBackendMessage>>::Error>,
    S::PortalStore: PortalStore,
{
    fn portal_store(&self) -> &MemPortalStore<DfStatement> {
        self.portal_store()
            .as_any()
            .downcast_ref::<MemPortalStore<DfStatement>>()
            .expect("portal store is not MemPortalStore<DfStatement>")
    }

    async fn send_message(&mut self, item: PgWireBackendMessage) -> PgWireResult<()> {
        use futures::SinkExt;
        self.send(item).await.map_err(PgWireError::from)
    }
}

#[async_trait]
pub trait QueryHook: Send + Sync {
    /// called in simple query handler to return response directly
    async fn handle_simple_query(
        &self,
        statement: &Statement,
        session_context: &SessionContext,
        client: &mut dyn HookClient,
    ) -> Option<PgWireResult<Response>>;

    /// called at extended query parse phase, for generating `LogicalPlan`from statement
    async fn handle_extended_parse_query(
        &self,
        sql: &Statement,
        session_context: &SessionContext,
        client: &(dyn ClientInfo + Send + Sync),
    ) -> Option<PgWireResult<LogicalPlan>>;

    /// Whether the plan this hook returned (for `canonical_sql`) was already
    /// optimized. Lets the do_query path skip a redundant `state.optimize()`
    /// call. Default `false` is conservative — the caller will optimize the
    /// plan as if it had been freshly parsed.
    fn was_pre_optimized(&self, _canonical_sql: &str) -> bool {
        false
    }

    /// How many trailing placeholders this hook injected into the plan beyond
    /// the client's binds (see `extra_execute_params`). The Parse/Describe path
    /// hides exactly this many from the `ParameterDescription` so the client
    /// still sees only its own params. MUST equal the number of values
    /// `extra_execute_params` appends for the same statement.
    /// `None` when the AST was not retained past Parse — see
    /// [`crate::handlers::retains_ast`]; no such statement carries an
    /// injected placeholder, so the count is zero.
    fn injected_param_count(&self, _statement: Option<&Statement>) -> usize {
        0
    }

    /// Extra positional parameter values for placeholders this hook injected
    /// into the plan at parse time BEYOND the client's bound params (e.g. a
    /// fresh `now()` instant). Appended to the client's deserialized params
    /// before `replace_params_with_values`, so those placeholders resolve to a
    /// fresh value on every execute — correct even for reused (named) prepared
    /// statements, where the parse hook runs only once. Values are matched by
    /// placeholder id, so any surplus (a statement the hook did not inject into)
    /// is ignored; returning `[]` is the safe default.
    /// `None` when the statement's AST was not retained past Parse — see
    /// [`crate::handlers::retains_ast`]. Only bulk data statements arrive that
    /// way, and none of them can carry an injected placeholder.
    fn extra_execute_params(&self, _statement: Option<&Statement>) -> Vec<ScalarValue> {
        Vec::new()
    }

    /// called at extended query execute phase, for query execution.
    ///
    /// `statement` is `None` when the AST was not retained past Parse (see
    /// [`crate::handlers::retains_ast`]): a bulk INSERT/UPDATE/DELETE/COPY,
    /// whose AST is the single largest thing a prepared statement can pin. A
    /// hook that dispatches on statement kind must decline on `None` — the
    /// kinds those hooks care about (SET/SHOW/DECLARE/transaction control) are
    /// always retained.
    async fn handle_extended_query(
        &self,
        statement: Option<&Statement>,
        logical_plan: &LogicalPlan,
        params: &ParamValues,
        session_context: &SessionContext,
        client: &mut dyn HookClient,
    ) -> Option<PgWireResult<Response>>;
}
