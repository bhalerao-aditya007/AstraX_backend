use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use ::document::{
    document::{image::Image, text::Text, voice::Voice},
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::{
        anpr::AnprProcessor,
        asr::AsrProcessor,
        base::DocumentProcessor,
        financial::FinancialProcessor,
        fir_ocr::FirOcrProcessor,
        ner::NerProcessor,
        ocr::OcrProcessor,
        yolo::YoloProcessor,
    },
};
use orm::entity::document as doc_entity;
use sea_orm::EntityTrait;
use uuid::Uuid;

use super::{AppError, AppState, MessageResponse};

// ---------------------------------------------------------------------------
// Handlers — Direct Model Trigger Endpoints
// ---------------------------------------------------------------------------

/// POST /api/models/fir-ocr/{doc_id}
///
/// Manually runs the FIR Key Information Extraction pipeline on a specific
/// image document.
async fn trigger_fir_ocr_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let ner = Arc::new(NerProcessor::with_default_url());
    let processor = FirOcrProcessor::new(gradio, ner);

    let image_doc = Image::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&image_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("FIR OCR completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/anpr/{doc_id}
async fn trigger_anpr_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let processor = AnprProcessor::new(gradio);

    let image_doc = Image::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&image_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("ANPR completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/yolo/{doc_id}
async fn trigger_yolo_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let processor = YoloProcessor::new(gradio);

    let image_doc = Image::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&image_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("YOLO detection completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/asr/{doc_id}
async fn trigger_asr_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let ner = Arc::new(NerProcessor::with_default_url());
    let processor = AsrProcessor::new(gradio, ner);

    let voice_doc = Voice::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&voice_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("ASR transcription completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/ner/{doc_id}
async fn trigger_ner_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let processor = NerProcessor::with_default_url();

    let text_doc = Text::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&text_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("NER extraction completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/ocr/{doc_id}
async fn trigger_ocr_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let ner = Arc::new(NerProcessor::with_default_url());
    let processor = OcrProcessor::with_default_url(ner);

    let image_doc = Image::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&image_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("Handwritten OCR completed for document {}", doc_id),
        }),
    ))
}

/// POST /api/models/financial/{doc_id}
async fn trigger_financial_handler(
    State(manager): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();
    let store = manager.storage();

    let doc_model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let processor = FinancialProcessor::new(gradio);

    let text_doc = Text::new(doc_id, doc_model.object_key.clone(), store.clone());
    processor.process(&text_doc, db).await?;

    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: format!("Financial analysis completed for document {}", doc_id),
        }),
    ))
}

// ---------------------------------------------------------------------------
// Router constructor
// ---------------------------------------------------------------------------

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/models/fir-ocr/{doc_id}", post(trigger_fir_ocr_handler))
        .route("/api/models/anpr/{doc_id}", post(trigger_anpr_handler))
        .route("/api/models/yolo/{doc_id}", post(trigger_yolo_handler))
        .route("/api/models/asr/{doc_id}", post(trigger_asr_handler))
        .route("/api/models/ner/{doc_id}", post(trigger_ner_handler))
        .route("/api/models/ocr/{doc_id}", post(trigger_ocr_handler))
        .route(
            "/api/models/financial/{doc_id}",
            post(trigger_financial_handler),
        )
        .with_state(state)
}
