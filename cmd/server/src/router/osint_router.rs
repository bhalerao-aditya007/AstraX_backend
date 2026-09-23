// cmd/server/src/router/osint_router.rs
//
// Axum routes for OSINT enrichment and retrieval.
// Three endpoints:
//   POST /api/osint/enrich/:doc_id    — enrich one document
//   POST /api/cases/:case_id/osint/enrich — enrich all documents in a case
//   GET  /api/cases/:case_id/osint    — read aggregated OSINT findings

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use document::{
    document::{
        base::Document,
        image::Image,
        text::Text,
        video::Video,
    },
    errors::DocumentErrors,
    processors::{base::DocumentProcessor, osint::OsintProcessor},
};
use orm::entity::{
    document as doc_entity,
    sea_orm_active_enums::DocumentType,
};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
use uuid::Uuid;

use super::{AppError, AppState};

// ── Helpers ─────────────────────────────────────────────────────────────

async fn enrich_single_doc(manager: &AppState, doc_id: Uuid) -> Result<usize, DocumentErrors> {
    let db = manager.db();
    let store = manager.storage();

    let model = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Document {} not found", doc_id)))?;

    let processor = OsintProcessor::from_env();
    let key = model.object_key.clone();

    match model.r#type {
        DocumentType::Image => {
            let img = Image::new(doc_id, key, store.clone());
            processor.process(&img, db).await?;
        }
        DocumentType::Video => {
            let vid = Video::new(doc_id, key, store.clone());
            processor.process(&vid, db).await?;
        }
        _ => {
            let txt = Text::new(doc_id, key, store.clone());
            processor.process(&txt, db).await?;
        }
    }

    // Read back finding count
    let refreshed = doc_entity::Entity::find_by_id(doc_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    Ok(refreshed
        .and_then(|d| d.extracted_information)
        .and_then(|i| i.get("osint").and_then(|o| o.as_array()).map(|a| a.len()))
        .unwrap_or(0))
}

// ── Handlers ────────────────────────────────────────────────────────────

/// POST /api/osint/enrich/:doc_id
async fn enrich_document_handler(
    State(state): State<AppState>,
    Path(doc_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let n = enrich_single_doc(&state, doc_id).await?;
    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "document_id": doc_id,
            "findings": n,
            "message": format!("OSINT enrichment complete — {} findings", n),
        })),
    ))
}

/// POST /api/cases/:case_id/osint/enrich
async fn enrich_case_handler(
    State(state): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = state.db();

    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::CaseId.eq(case_id))
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut total = 0usize;
    for d in &docs {
        total += enrich_single_doc(&state, d.id).await.unwrap_or(0);
    }

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "case_id": case_id,
            "documents_scanned": docs.len(),
            "total_findings": total,
        })),
    ))
}

/// GET /api/cases/:case_id/osint
async fn get_case_osint_handler(
    State(state): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = state.db();

    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::CaseId.eq(case_id))
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut findings = Vec::<serde_json::Value>::new();
    for d in &docs {
        if let Some(arr) = d
            .extracted_information
            .as_ref()
            .and_then(|i| i.get("osint"))
            .and_then(|o| o.as_array())
        {
            for f in arr {
                let mut f = f.clone();
                if let Some(m) = f.as_object_mut() {
                    m.insert(
                        "source_document".into(),
                        serde_json::json!(d.title),
                    );
                    m.insert(
                        "document_id".into(),
                        serde_json::json!(d.id),
                    );
                }
                findings.push(f);
            }
        }
    }

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "case_id": case_id,
            "total": findings.len(),
            "findings": findings,
        })),
    ))
}

// ── Router ──────────────────────────────────────────────────────────────

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route(
            "/api/osint/enrich/{doc_id}",
            post(enrich_document_handler),
        )
        .route(
            "/api/cases/{case_id}/osint/enrich",
            post(enrich_case_handler),
        )
        .route(
            "/api/cases/{case_id}/osint",
            get(get_case_osint_handler),
        )
        .with_state(state)
}
