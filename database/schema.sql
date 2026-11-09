-- =====================================================================
-- Adaptive Recommender — esquema relacional
-- SQL compatível com SQLite e, com ajustes mínimos de tipos, PostgreSQL
-- (INTEGER PRIMARY KEY -> GENERATED ALWAYS AS IDENTITY; TEXT JSON -> JSONB).
--
-- Camadas:
--   * Dados brutos/operacionais: customers, products, orders, order_items, interactions
--   * Dados processados:         customer_features
--   * Resultados de modelos:     model_versions, cluster_*, bayesian_*, recommendation_*,
--                                expert_weights, evaluation_*
--   * Auditoria:                 event_log, bayesian_update_log
-- =====================================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id                 INTEGER PRIMARY KEY,
    name                        TEXT NOT NULL UNIQUE,
    city                        TEXT NOT NULL,
    state                       TEXT NOT NULL CHECK (length(state) = 2),
    registration_date           TIMESTAMP NOT NULL,
    store_size                  TEXT NOT NULL CHECK (store_size IN ('Pequeno', 'Médio', 'Grande')),
    business_segment            TEXT NOT NULL,
    revenue_band                TEXT NOT NULL,
    declared_categories         TEXT,
    declared_price_band         TEXT CHECK (declared_price_band IN ('Econômica', 'Intermediária', 'Premium')),
    declared_purchase_frequency TEXT,
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT NOT NULL,
    target_audience TEXT NOT NULL,
    price_band      TEXT NOT NULL CHECK (price_band IN ('Econômica', 'Intermediária', 'Premium')),
    price           REAL NOT NULL CHECK (price > 0),
    cost            REAL NOT NULL CHECK (cost >= 0),
    margin          REAL NOT NULL CHECK (margin >= -1 AND margin <= 1),
    color           TEXT,
    size_range      TEXT,
    material        TEXT,
    season          TEXT CHECK (season IN ('Verão', 'Inverno', 'Atemporal')),
    collection      TEXT,
    launch_date     TIMESTAMP NOT NULL,
    stock           INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    description     TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);
CREATE INDEX IF NOT EXISTS ix_products_category ON products (category);

CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (customer_id) ON DELETE RESTRICT,
    order_date  TIMESTAMP NOT NULL,
    channel     TEXT,
    source      TEXT NOT NULL DEFAULT 'synthetic' CHECK (source IN ('synthetic', 'app'))
);
CREATE INDEX IF NOT EXISTS ix_orders_customer ON orders (customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_date ON orders (order_date);

CREATE TABLE IF NOT EXISTS order_items (
    item_id     INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    product_id  INTEGER NOT NULL REFERENCES products (product_id) ON DELETE RESTRICT,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    unit_price  REAL NOT NULL CHECK (unit_price > 0),
    discount    REAL NOT NULL DEFAULT 0 CHECK (discount >= 0 AND discount < 1),
    total_value REAL NOT NULL CHECK (total_value >= 0)
);
CREATE INDEX IF NOT EXISTS ix_items_order ON order_items (order_id);
CREATE INDEX IF NOT EXISTS ix_items_product ON order_items (product_id);

-- Execuções do recomendador (auditoria do que foi exibido e por quê)
CREATE TABLE IF NOT EXISTS recommendation_runs (
    run_id              INTEGER PRIMARY KEY,
    customer_id         INTEGER NOT NULL REFERENCES customers (customer_id),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    as_of               TIMESTAMP NOT NULL,
    gating_method       TEXT NOT NULL,
    config_json         TEXT,
    model_versions_json TEXT,
    justification       TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_customer ON recommendation_runs (customer_id, created_at);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id      INTEGER PRIMARY KEY,
    run_id                 INTEGER NOT NULL REFERENCES recommendation_runs (run_id) ON DELETE CASCADE,
    product_id             INTEGER NOT NULL REFERENCES products (product_id),
    rank                   INTEGER NOT NULL CHECK (rank >= 1),
    final_score            REAL NOT NULL,
    relevance_score        REAL NOT NULL CHECK (relevance_score >= 0 AND relevance_score <= 1),
    conversion_probability REAL CHECK (conversion_probability IS NULL OR (conversion_probability >= 0 AND conversion_probability <= 1)),
    contributions_json     TEXT,
    reasons                TEXT,
    UNIQUE (run_id, rank)
);
CREATE INDEX IF NOT EXISTS ix_recs_run ON recommendations (run_id);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id    INTEGER PRIMARY KEY,
    session_id        INTEGER NOT NULL,
    customer_id       INTEGER NOT NULL REFERENCES customers (customer_id),
    product_id        INTEGER NOT NULL REFERENCES products (product_id),
    model_name        TEXT NOT NULL,
    interaction_type  TEXT NOT NULL CHECK (interaction_type IN ('impression', 'click', 'interest', 'add_to_cart', 'purchase', 'rejection')),
    timestamp         TIMESTAMP NOT NULL,
    context           TEXT,
    recommendation_id INTEGER REFERENCES recommendations (recommendation_id),
    source            TEXT NOT NULL DEFAULT 'synthetic' CHECK (source IN ('synthetic', 'app'))
);
CREATE INDEX IF NOT EXISTS ix_inter_customer ON interactions (customer_id, timestamp);
CREATE INDEX IF NOT EXISTS ix_inter_session ON interactions (session_id);
CREATE INDEX IF NOT EXISTS ix_inter_product ON interactions (product_id);

-- Registro e versionamento de modelos
CREATE TABLE IF NOT EXISTS models (
    model_name  TEXT PRIMARY KEY,
    model_type  TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id    INTEGER PRIMARY KEY,
    model_name    TEXT NOT NULL REFERENCES models (model_name),
    version       INTEGER NOT NULL CHECK (version >= 1),
    trained_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    train_cutoff  TIMESTAMP NOT NULL,
    params_json   TEXT,
    metrics_json  TEXT,
    artifact_path TEXT,
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    UNIQUE (model_name, version)
);
CREATE INDEX IF NOT EXISTS ix_versions_model ON model_versions (model_name, status);

CREATE TABLE IF NOT EXISTS expert_weights (
    weight_id    INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES recommendation_runs (run_id) ON DELETE CASCADE,
    customer_id  INTEGER NOT NULL REFERENCES customers (customer_id),
    expert       TEXT NOT NULL,
    weight       REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    method       TEXT NOT NULL,
    signals_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_weights_customer ON expert_weights (customer_id, run_id);

-- Estado bayesiano (Beta) por cliente × categoria × estágio do funil
CREATE TABLE IF NOT EXISTS bayesian_estimates (
    customer_id      INTEGER NOT NULL REFERENCES customers (customer_id),
    category         TEXT NOT NULL,
    stage            TEXT NOT NULL CHECK (stage IN ('interest', 'conversion')),
    alpha_prior      REAL NOT NULL CHECK (alpha_prior > 0),
    beta_prior       REAL NOT NULL CHECK (beta_prior > 0),
    alpha            REAL NOT NULL CHECK (alpha > 0),
    beta             REAL NOT NULL CHECK (beta > 0),
    n_events         INTEGER NOT NULL DEFAULT 0,
    updated_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version_id INTEGER REFERENCES model_versions (version_id),
    PRIMARY KEY (customer_id, category, stage)
);

CREATE TABLE IF NOT EXISTS bayesian_update_log (
    log_id         INTEGER PRIMARY KEY,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    customer_id    INTEGER NOT NULL REFERENCES customers (customer_id),
    category       TEXT NOT NULL,
    stage          TEXT NOT NULL,
    interaction_id INTEGER REFERENCES interactions (interaction_id),
    alpha_before   REAL NOT NULL,
    beta_before    REAL NOT NULL,
    alpha_after    REAL NOT NULL,
    beta_after     REAL NOT NULL,
    mean_after     REAL NOT NULL CHECK (mean_after >= 0 AND mean_after <= 1),
    ci_low         REAL,
    ci_high        REAL,
    evidence_s     REAL NOT NULL DEFAULT 0,
    evidence_f     REAL NOT NULL DEFAULT 0,
    reason         TEXT
);
CREATE INDEX IF NOT EXISTS ix_blog_customer ON bayesian_update_log (customer_id, category, stage);

-- Segmentação
CREATE TABLE IF NOT EXISTS cluster_runs (
    cluster_run_id INTEGER PRIMARY KEY,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    as_of          TIMESTAMP NOT NULL,
    algorithm      TEXT NOT NULL,
    k              INTEGER NOT NULL CHECK (k >= 2),
    silhouette     REAL,
    params_json    TEXT,
    diagnostics_json TEXT,
    status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived'))
);

CREATE TABLE IF NOT EXISTS cluster_profiles (
    cluster_run_id INTEGER NOT NULL REFERENCES cluster_runs (cluster_run_id) ON DELETE CASCADE,
    cluster_id     INTEGER NOT NULL,
    label          TEXT NOT NULL,
    description    TEXT,
    size           INTEGER NOT NULL,
    centroid_json  TEXT,
    PRIMARY KEY (cluster_run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS customer_clusters (
    cluster_run_id INTEGER NOT NULL REFERENCES cluster_runs (cluster_run_id) ON DELETE CASCADE,
    customer_id    INTEGER NOT NULL REFERENCES customers (customer_id),
    cluster_id     INTEGER NOT NULL,
    cluster_label  TEXT NOT NULL,
    PRIMARY KEY (cluster_run_id, customer_id)
);

-- Dados processados (features comportamentais, recalculadas incrementalmente)
CREATE TABLE IF NOT EXISTS customer_features (
    customer_id   INTEGER PRIMARY KEY REFERENCES customers (customer_id),
    as_of         TIMESTAMP NOT NULL,
    features_json TEXT NOT NULL,
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Avaliação experimental
CREATE TABLE IF NOT EXISTS evaluation_runs (
    eval_run_id   INTEGER PRIMARY KEY,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    protocol_json TEXT NOT NULL,
    summary_json  TEXT,
    report_path   TEXT,
    status        TEXT NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id     INTEGER PRIMARY KEY,
    eval_run_id   INTEGER NOT NULL REFERENCES evaluation_runs (eval_run_id) ON DELETE CASCADE,
    stage         TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    segment_type  TEXT NOT NULL,
    segment_value TEXT NOT NULL,
    metric        TEXT NOT NULL,
    k             INTEGER,
    value         REAL,
    ci_low        REAL,
    ci_high       REAL,
    n_customers   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_evalres_run ON evaluation_results (eval_run_id, stage, metric);

-- Auditoria geral
CREATE TABLE IF NOT EXISTS event_log (
    event_id     INTEGER PRIMARY KEY,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type   TEXT NOT NULL,
    entity       TEXT,
    entity_id    TEXT,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_type ON event_log (event_type, created_at);

-- Visão achatada dos itens de pedido (inclui categoria, conforme especificação)
CREATE VIEW IF NOT EXISTS v_order_items_flat AS
SELECT oi.item_id, o.order_id, o.customer_id, oi.product_id, o.order_date,
       oi.quantity, oi.unit_price, oi.total_value, oi.discount, p.category, o.source
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id;
