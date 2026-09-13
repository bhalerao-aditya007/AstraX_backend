use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use document::{
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::{
        historical_analyzer::HistoricalAnalyzer,
        summarizer::SummarizerProcessor,
    },
};
use orm::entity::{case, document as doc_entity, gnn_link_prediction};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
use serde::Serialize;
use uuid::Uuid;

use super::{AppError, AppState};

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct EntitySummary {
    pub entity_id: String,
    pub entity_type: String,
    pub display_name: String,
    pub source_documents: Vec<String>,
    pub connections_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossCaseConnection {
    pub entity_id: String,
    pub display_name: String,
    pub entity_type: String,
    pub appearing_in_cases: Vec<String>,
    pub total_appearances: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseEntitiesResponse {
    pub case_id: String,
    pub case_name: String,
    pub total_entities: usize,
    pub entities: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CrossCaseResponse {
    pub total_cross_case_entities: usize,
    pub connections: Vec<CrossCaseConnection>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/cases/{id}/analyze-historical
///
/// Runs the full historical analysis pipeline for a case:
/// Entity resolution → GNN inference → MO matching → Theory generation
async fn analyze_historical_handler(
    State(manager): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    // Verify case exists
    let _case = case::Entity::find_by_id(case_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Case {} not found", case_id)))?;

    let gradio = Arc::new(GradioClient::from_env());
    let gnn_url = std::env::var("GNN_SERVICE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());

    let analyzer = HistoricalAnalyzer::new(gradio, gnn_url);
    let report = analyzer.analyze_case(case_id, db).await?;

    tracing::info!(
        "Historical analysis completed for case {}: {} entities, {} leads, {} theories",
        case_id,
        report.total_entities,
        report.priority_leads.len(),
        report.theories.len(),
    );

    Ok((StatusCode::OK, Json(report)))
}

/// GET /api/cases/{id}/entities
///
/// Lists all resolved entities for a case by scanning extracted information
/// from all case documents.
async fn get_case_entities_handler(
    State(manager): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    let case_model = case::Entity::find_by_id(case_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Case {} not found", case_id)))?;

    // Fetch case documents with extracted information
    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::CaseId.eq(case_id))
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut entities: Vec<EntitySummary> = Vec::new();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for doc in &docs {
        if let Some(info) = &doc.extracted_information {
            extract_entities_from_json(
                info,
                &doc.title,
                &mut entities,
                &mut seen,
            );
        }
    }

    let response = CaseEntitiesResponse {
        case_id: case_id.to_string(),
        case_name: case_model.name,
        total_entities: entities.len(),
        entities,
    };

    Ok((StatusCode::OK, Json(response)))
}

/// GET /api/entities/{id}/connections
///
/// Returns all GNN-predicted and evidentiary connections for a specific entity.
async fn get_entity_connections_handler(
    State(manager): State<AppState>,
    Path(entity_id): Path<String>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    // Find GNN predictions involving this entity
    let predictions = gnn_link_prediction::Entity::find()
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut connections = Vec::new();
    for pred in predictions {
        if pred.target_entity_id == entity_id || pred.candidate_entity_id == entity_id {
            let other_id = if pred.target_entity_id == entity_id {
                &pred.candidate_entity_id
            } else {
                &pred.target_entity_id
            };

            connections.push(serde_json::json!({
                "connected_entity_id": other_id,
                "edge_type": pred.predicted_edge_type,
                "probability": pred.link_probability,
                "is_hypothesis": pred.is_hypothesis_flagged,
                "recommendation": pred.recommendation,
            }));
        }
    }

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "entity_id": entity_id,
            "total_connections": connections.len(),
            "connections": connections,
        })),
    ))
}

/// GET /api/analysis/cross-case-connections
///
/// Finds entities that appear across multiple cases, indicating potential
/// criminal network connections.
async fn cross_case_connections_handler(
    State(manager): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    // Fetch all documents with extracted information, grouped by case
    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    // Map: entity_id -> Set of case_ids where it appears
    let mut entity_cases: std::collections::HashMap<
        String,
        (String, String, std::collections::HashSet<Uuid>),
    > = std::collections::HashMap::new();

    for doc in &docs {
        if let Some(info) = &doc.extracted_information {
            let case_id = doc.case_id;
            extract_entity_case_mapping(info, case_id, &mut entity_cases);
        }
    }

    // Fetch case names for display
    let cases = case::Entity::find()
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;
    let case_names: std::collections::HashMap<Uuid, String> = cases
        .into_iter()
        .map(|c| (c.id, c.name))
        .collect();

    // Filter to entities appearing in 2+ cases
    let mut connections: Vec<CrossCaseConnection> = entity_cases
        .into_iter()
        .filter(|(_, (_, _, cases))| cases.len() >= 2)
        .map(|(entity_id, (display_name, entity_type, case_ids))| {
            let appearing_in_cases: Vec<String> = case_ids
                .iter()
                .map(|cid| {
                    case_names
                        .get(cid)
                        .cloned()
                        .unwrap_or_else(|| cid.to_string())
                })
                .collect();
            let total = appearing_in_cases.len();
            CrossCaseConnection {
                entity_id,
                display_name,
                entity_type,
                appearing_in_cases,
                total_appearances: total,
            }
        })
        .collect();

    connections.sort_by(|a, b| b.total_appearances.cmp(&a.total_appearances));

    let response = CrossCaseResponse {
        total_cross_case_entities: connections.len(),
        connections,
    };

    Ok((StatusCode::OK, Json(response)))
}

/// POST /api/cases/{id}/fact-sheet
///
/// Generates or regenerates the case fact sheet summary.
async fn generate_fact_sheet_handler(
    State(manager): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    let _case = case::Entity::find_by_id(case_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Case {} not found", case_id)))?;

    // Collect all extracted information for the case
    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::CaseId.eq(case_id))
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let case_data: Vec<serde_json::Value> = docs
        .iter()
        .filter_map(|d| d.extracted_information.clone())
        .collect();

    let gradio = Arc::new(GradioClient::from_env());
    let summarizer = SummarizerProcessor::new(gradio);

    let fact_sheet = summarizer
        .summarize_case(serde_json::json!({
            "case_id": case_id.to_string(),
            "documents": case_data,
        }))
        .await?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "case_id": case_id.to_string(),
            "fact_sheet": fact_sheet,
            "generated_at": chrono::Utc::now().to_rfc3339(),
        })),
    ))
}

/// GET /api/cases/{id}/theories
///
/// Returns all generated theories for a case (from GNN + historical analysis).
async fn get_case_theories_handler(
    State(manager): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let db = manager.db();

    let _case = case::Entity::find_by_id(case_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| DocumentErrors::NotFound(format!("Case {} not found", case_id)))?;

    // Since theory table may not have ORM entities generated yet, we query
    // via the analysis pipeline's cached results. For now, run a fresh analysis
    // and return the theories portion.
    //
    // In production, theories would be persisted to the `theory` table and
    // queried directly. This endpoint provides the real-time computation path.
    let gradio = Arc::new(GradioClient::from_env());
    let gnn_url = std::env::var("GNN_SERVICE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());

    let analyzer = HistoricalAnalyzer::new(gradio, gnn_url);
    let report = analyzer.analyze_case(case_id, db).await?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "case_id": case_id.to_string(),
            "total_theories": report.theories.len(),
            "theories": report.theories,
            "priority_leads": report.priority_leads,
        })),
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_entities_from_json(
    info: &serde_json::Value,
    doc_title: &str,
    entities: &mut Vec<EntitySummary>,
    seen: &mut std::collections::HashMap<String, usize>,
) {
    if let serde_json::Value::Object(map) = info {
        for (key, val) in map {
            let k_lower = key.to_lowercase();
            if let Some(s) = val.as_str() {
                let s_clean = s.trim();
                if s_clean.is_empty() {
                    continue;
                }

                let (entity_id, entity_type) = if k_lower.contains("name")
                    || k_lower.contains("suspect")
                    || k_lower.contains("accused")
                    || k_lower.contains("person")
                {
                    (
                        format!("ENT-PERSON-{}", s_clean.to_uppercase().replace(' ', "_")),
                        "person".to_string(),
                    )
                } else if k_lower.contains("phone") || k_lower.contains("mobile") {
                    (
                        format!("ENT-PHONE-{}", s_clean.replace(' ', "")),
                        "phone".to_string(),
                    )
                } else if k_lower.contains("account") || k_lower.contains("bank") {
                    (
                        format!("ENT-ACC-{}", s_clean.replace(' ', "")),
                        "financial_account".to_string(),
                    )
                } else if k_lower.contains("vehicle")
                    || k_lower.contains("plate")
                    || k_lower.contains("car")
                {
                    (
                        format!("ENT-OBJ-{}", s_clean.to_uppercase().replace(' ', "_")),
                        "vehicle".to_string(),
                    )
                } else if k_lower.contains("weapon") || k_lower.contains("object") {
                    (
                        format!("ENT-OBJ-{}", s_clean.to_uppercase().replace(' ', "_")),
                        "object".to_string(),
                    )
                } else {
                    continue;
                };

                if let Some(&idx) = seen.get(&entity_id) {
                    entities[idx]
                        .source_documents
                        .push(doc_title.to_string());
                    entities[idx].connections_count += 1;
                } else {
                    seen.insert(entity_id.clone(), entities.len());
                    entities.push(EntitySummary {
                        entity_id,
                        entity_type,
                        display_name: s_clean.to_string(),
                        source_documents: vec![doc_title.to_string()],
                        connections_count: 0,
                    });
                }
            } else if val.is_array() {
                if let Some(arr) = val.as_array() {
                    for item in arr {
                        extract_entities_from_json(item, doc_title, entities, seen);
                    }
                }
            } else if val.is_object() {
                extract_entities_from_json(val, doc_title, entities, seen);
            }
        }
    }
}

fn extract_entity_case_mapping(
    info: &serde_json::Value,
    case_id: Uuid,
    entity_cases: &mut std::collections::HashMap<
        String,
        (String, String, std::collections::HashSet<Uuid>),
    >,
) {
    if let serde_json::Value::Object(map) = info {
        for (key, val) in map {
            let k_lower = key.to_lowercase();
            if let Some(s) = val.as_str() {
                let s_clean = s.trim();
                if s_clean.is_empty() {
                    continue;
                }

                let (entity_id, entity_type) = if k_lower.contains("name")
                    || k_lower.contains("suspect")
                    || k_lower.contains("accused")
                    || k_lower.contains("person")
                {
                    (
                        format!("ENT-PERSON-{}", s_clean.to_uppercase().replace(' ', "_")),
                        "person".to_string(),
                    )
                } else if k_lower.contains("phone") || k_lower.contains("mobile") {
                    (
                        format!("ENT-PHONE-{}", s_clean.replace(' ', "")),
                        "phone".to_string(),
                    )
                } else if k_lower.contains("vehicle") || k_lower.contains("plate") {
                    (
                        format!("ENT-OBJ-{}", s_clean.to_uppercase().replace(' ', "_")),
                        "vehicle".to_string(),
                    )
                } else {
                    continue;
                };

                entity_cases
                    .entry(entity_id)
                    .or_insert_with(|| {
                        (s_clean.to_string(), entity_type, std::collections::HashSet::new())
                    })
                    .2
                    .insert(case_id);
            } else if val.is_array() {
                if let Some(arr) = val.as_array() {
                    for item in arr {
                        extract_entity_case_mapping(item, case_id, entity_cases);
                    }
                }
            } else if val.is_object() {
                extract_entity_case_mapping(val, case_id, entity_cases);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Router constructor
// ---------------------------------------------------------------------------

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route(
            "/api/cases/{id}/analyze-historical",
            post(analyze_historical_handler),
        )
        .route("/api/cases/{id}/entities", get(get_case_entities_handler))
        .route(
            "/api/entities/{id}/connections",
            get(get_entity_connections_handler),
        )
        .route(
            "/api/analysis/cross-case-connections",
            get(cross_case_connections_handler),
        )
        .route(
            "/api/cases/{id}/fact-sheet",
            post(generate_fact_sheet_handler),
        )
        .route(
            "/api/cases/{id}/theories",
            get(get_case_theories_handler),
        )
        .with_state(state)
}
