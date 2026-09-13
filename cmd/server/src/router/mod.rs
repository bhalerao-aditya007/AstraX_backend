pub mod case_router;
pub mod document_router;
pub mod gnn_router;
pub mod analysis_router;
pub mod model_router;

use std::sync::Arc;
use axum::{
    Json, Router,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use document::document_manager::DocumentManager;
use serde::Serialize;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

pub type AppState = Arc<DocumentManager>;

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: String,
}

pub struct AppError(pub document::errors::DocumentErrors);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match &self.0 {
            document::errors::DocumentErrors::NotFound(msg) => {
                (StatusCode::NOT_FOUND, msg.clone())
            }
            document::errors::DocumentErrors::ValidationError(msg) => {
                (StatusCode::BAD_REQUEST, msg.clone())
            }
            document::errors::DocumentErrors::StorageError(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, msg.clone())
            }
            document::errors::DocumentErrors::NetworkError(e) => {
                (StatusCode::BAD_GATEWAY, e.to_string())
            }
            document::errors::DocumentErrors::DatabaseError(e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            }
        };

        tracing::error!("Request error: {}", message);
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

impl From<document::errors::DocumentErrors> for AppError {
    fn from(err: document::errors::DocumentErrors) -> Self {
        AppError(err)
    }
}

/// GET /health
async fn health_check_handler() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Build the unified router mounting all subrouters and middlewares
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check_handler))
        .route("/api/health", get(health_check_handler))
        .merge(document_router::create_router(state.clone()))
        .merge(case_router::create_router(state.clone()))
        .merge(gnn_router::create_router(state.clone()))
        .merge(analysis_router::create_router(state.clone()))
        .merge(model_router::create_router(state))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}
