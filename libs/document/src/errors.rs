use thiserror::Error;

#[derive(Debug, Error)]
pub enum DocumentErrors {
    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Database error: {0}")]
    DatabaseError(#[from] sea_orm::DbErr),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}
