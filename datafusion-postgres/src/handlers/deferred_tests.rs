use super::*;
use crate::auth::AuthManager;
use crate::hooks::permissions::PermissionsHook;
use pgwire::api::DefaultClient;

#[tokio::test]
async fn bulk_retention_preserves_descriptions() -> Result<(), Box<dyn std::error::Error>> {
    let context = Arc::new(SessionContext::new());
    context
        .sql("CREATE TABLE t (id INT, name TEXT)")
        .await?
        .collect()
        .await?;
    let parser = Parser {
        session_context: context,
        sql_parser: PostgresCompatibilityParser::new(),
        query_hooks: vec![],
    };
    let client = DefaultClient::<ParsedStatement>::new("127.0.0.1:0".parse()?, false);
    for (sql, bulk, types) in [
        (
            "INSERT INTO t VALUES ($1, $2), ($3, $4)",
            true,
            vec![Type::INT4, Type::TEXT, Type::INT4, Type::TEXT],
        ),
        (
            "UPDATE t SET name = $1 WHERE id = $2",
            true,
            vec![Type::TEXT, Type::INT4],
        ),
        ("DELETE FROM t WHERE id = $1", true, vec![Type::INT4]),
        ("SELECT id FROM t WHERE id = $1", false, vec![Type::INT4]),
        ("SELECT COUNT(*) AS count FROM t", false, vec![]),
    ] {
        let statement = parser
            .parse_sql(&client, sql, &[])
            .await?
            .expect("statement");
        let (ast, retained) = statement.1.as_ref().expect("planned statement");
        assert_eq!(ast.is_none(), bulk, "AST retention: {sql}");
        assert_eq!(retained.held().is_none(), bulk, "plan retention: {sql}");
        assert_eq!(parser.get_parameter_types(&statement)?, types, "{sql}");
        assert_eq!(
            parser.get_result_schema(&statement, None)?.is_empty(),
            bulk,
            "{sql}"
        );
        let (rebuilt_ast, rebuilt_plan) = parser.rebuild(&statement.0, &client).await?;
        assert_eq!(is_bulk_data(&rebuilt_ast), bulk);
        assert_eq!(parameter_wire_types(&rebuilt_plan)?, types);
    }
    Ok(())
}

#[tokio::test]
async fn deferred_execution_checks_permissions_on_every_execute()
-> Result<(), Box<dyn std::error::Error>> {
    let context = Arc::new(SessionContext::new());
    context
        .sql("CREATE TABLE t (id INT)")
        .await?
        .collect()
        .await?;
    let mut hooks = default_query_hooks();
    hooks.push(Arc::new(PermissionsHook::new(Arc::new(
        AuthManager::default(),
    ))));
    let factory = Arc::new(HandlerFactory::new_with_hooks(context, hooks));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let server = tokio::spawn(async move {
        let mut connections = tokio::task::JoinSet::new();
        for _ in 0..2 {
            let (socket, _) = listener.accept().await.unwrap();
            let factory = Arc::clone(&factory);
            connections.spawn(async move {
                pgwire::tokio::process_socket(socket, None, factory)
                    .await
                    .unwrap();
            });
        }
        while let Some(result) = connections.join_next().await {
            result.unwrap();
        }
    });
    tokio::task::spawn_blocking(move || {
        for (user, allowed) in [("anonymous", false), ("postgres", true)] {
            let mut client = postgres::Config::new()
                .host("127.0.0.1")
                .port(port)
                .user(user)
                .dbname("datafusion")
                .connect_timeout(std::time::Duration::from_secs(5))
                .connect(postgres::NoTls)
                .unwrap();
            let statement = client.prepare("INSERT INTO t VALUES ($1)").unwrap();
            for value in [1_i32, 2] {
                let result = client.execute(&statement, &[&value]);
                if allowed {
                    assert_eq!(result.unwrap(), 1);
                } else {
                    assert_eq!(result.unwrap_err().code().unwrap().code(), "42501");
                }
            }
            if allowed {
                let count: i64 = client
                    .query_one("SELECT COUNT(*) FROM t", &[])
                    .unwrap()
                    .get(0);
                assert_eq!(count, 2, "denied executions must not insert rows");
            }
        }
    })
    .await?;
    server.await?;
    Ok(())
}

#[test]
fn synonym_rewrite_preserves_non_ascii_prefixes() {
    for sql in ["ééé", "日abc", "🙂ab", "SELECT 'é'", "ABORTé"] {
        assert_eq!(rewrite_postgres_synonyms(sql), sql);
    }
}
