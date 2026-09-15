use std::collections::{HashMap, HashSet};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use document::{
    errors::DocumentErrors,
    processors::{base::DocumentProcessor, gnn::GnnProcessor},
};
use orm::entity::{case, document as doc_entity, gnn_link_prediction};
use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};
use serde::Serialize;
use uuid::Uuid;

use super::{AppError, AppState, MessageResponse};

// ---------------------------------------------------------------------------
// Cytoscape.js JSON DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct NodeData {
    pub id: String,
    pub label: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_score: Option<f64>,
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CytoscapeNode {
    pub data: NodeData,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeData {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: String,
    pub category: String, // "evidentiary" or "hypothesis"
    pub is_hypothesis: bool,
    pub style: String,    // "solid" or "dashed"
    pub color: String,    // "#64748b" for evidentiary, "#ef4444" or "#f59e0b" for hypothesis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recommendation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_document: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CytoscapeEdge {
    pub data: EdgeData,
}

#[derive(Debug, Clone, Serialize)]
pub struct CytoscapeElements {
    pub nodes: Vec<CytoscapeNode>,
    pub edges: Vec<CytoscapeEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphMeta {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub evidentiary_edges: usize,
    pub predicted_edges: usize,
    pub last_analyzed: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CytoscapeGraphResponse {
    pub status: String,
    pub case_id: Option<String>,
    pub meta: GraphMeta,
    pub elements: CytoscapeElements,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/gnn/trigger
///
/// Manually triggers the collective GNN inference pipeline across all documents
/// in the database using GnnProcessor::cron_func.
async fn trigger_gnn_handler(
    State(manager): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    tracing::info!("Received manual trigger request for GNN collective inference");
    // Read GNN microservice URL from environment
    let gnn_url = std::env::var("GNN_SERVICE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());

    let processor = GnnProcessor::new(gnn_url);

    processor
        .cron_func(manager.db(), manager.storage())
        .await?;

    tracing::info!("GNN collective inference triggered and executed successfully");
    Ok((
        StatusCode::OK,
        Json(MessageResponse {
            message: "GNN collective inference triggered and updated successfully".to_string(),
        }),
    ))
}

/// GET /api/cases/{case_id}/graph
///
/// Fetches case evidentiary elements combined with GNN hypothesis links formatted
/// directly for frontend Cytoscape.js canvas rendering.
async fn get_case_graph_handler(
    State(manager): State<AppState>,
    Path(case_id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    tracing::info!("Building Cytoscape graph for case_id={}", case_id);
    let db = manager.db();

    // Verify case exists
    let case_model = case::Entity::find_by_id(case_id)
        .one(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?
        .ok_or_else(|| {
            DocumentErrors::NotFound(format!("Case {} not found", case_id))
        })?;

    // Fetch documents for this case with extracted information
    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::CaseId.eq(case_id))
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut nodes_map: HashMap<String, NodeData> = HashMap::new();
    let mut edges: Vec<CytoscapeEdge> = Vec::new();
    let mut edge_counter = 1;

    // 1. Extract evidentiary nodes and edges from case documents
    for doc in &docs {
        if let Some(info) = &doc.extracted_information {
            let doc_title = doc.title.clone();
            extract_evidentiary_elements(
                info,
                &doc_title,
                &mut nodes_map,
                &mut edges,
                &mut edge_counter,
            );
        }
    }

    let evidentiary_count = edges.len();

    // 2. Fetch GNN hypothesis links
    let case_entity_ids: HashSet<String> = nodes_map.keys().cloned().collect();
    let predictions = gnn_link_prediction::Entity::find()
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut predicted_count = 0;
    for pred in predictions {
        // Link is relevant if either target or candidate is in this case's entities
        let target_in = case_entity_ids.contains(&pred.target_entity_id);
        let candidate_in = case_entity_ids.contains(&pred.candidate_entity_id);

        if target_in || candidate_in {
            // Ensure both nodes exist in the graph so Cytoscape can render the edge
            if !nodes_map.contains_key(&pred.target_entity_id) {
                nodes_map.insert(
                    pred.target_entity_id.clone(),
                    NodeData {
                        id: pred.target_entity_id.clone(),
                        label: format_label(&pred.target_entity_id),
                        r#type: pred.target_entity_type.clone(),
                        badge: Some("Associated Suspect".to_string()),
                        risk_score: None,
                        attributes: serde_json::Map::new(),
                    },
                );
            }

            if !nodes_map.contains_key(&pred.candidate_entity_id) {
                nodes_map.insert(
                    pred.candidate_entity_id.clone(),
                    NodeData {
                        id: pred.candidate_entity_id.clone(),
                        label: format_label(&pred.candidate_entity_id),
                        r#type: pred.candidate_entity_type.clone(),
                        badge: Some("Candidate Associate".to_string()),
                        risk_score: Some(pred.link_probability),
                        attributes: serde_json::Map::new(),
                    },
                );
            }

            let color = if pred.link_probability >= 0.70 {
                "#ef4444".to_string() // Bright Red (High Alert)
            } else {
                "#f59e0b".to_string() // Amber (Moderate Alert)
            };

            let label = format!(
                "{} ({:.1}%)",
                pred.predicted_edge_type,
                pred.link_probability * 100.0
            );

            edges.push(CytoscapeEdge {
                data: EdgeData {
                    id: format!("edge-gnn-pred-{}", pred.id),
                    source: pred.target_entity_id,
                    target: pred.candidate_entity_id,
                    label,
                    category: "hypothesis".to_string(),
                    is_hypothesis: true,
                    style: "dashed".to_string(),
                    color,
                    probability: Some(pred.link_probability),
                    recommendation: pred.recommendation,
                    source_document: None,
                },
            });
            predicted_count += 1;
        }
    }

    let nodes: Vec<CytoscapeNode> = nodes_map
        .into_values()
        .map(|data| CytoscapeNode { data })
        .collect();

    let total_nodes = nodes.len();
    let total_edges = edges.len();

    let response = CytoscapeGraphResponse {
        status: "success".to_string(),
        case_id: Some(case_model.name),
        meta: GraphMeta {
            total_nodes,
            total_edges,
            evidentiary_edges: evidentiary_count,
            predicted_edges: predicted_count,
            last_analyzed: chrono::Utc::now().to_rfc3339(),
        },
        elements: CytoscapeElements { nodes, edges },
    };

    tracing::info!(
        "Assembled Cytoscape graph for case_id={}: {} nodes, {} edges ({} evidentiary, {} predicted)",
        case_id, total_nodes, total_edges, evidentiary_count, predicted_count
    );

    Ok((StatusCode::OK, Json(response)))
}

/// GET /api/gnn/graph
///
/// Global multi-case graph endpoint returning all entities and all GNN hypothesis
/// links formatted for Cytoscape.js canvas rendering.
async fn get_global_graph_handler(
    State(manager): State<AppState>,
) -> Result<impl IntoResponse, AppError> {
    tracing::info!("Building global multi-case Cytoscape graph");
    let db = manager.db();

    let docs = doc_entity::Entity::find()
        .filter(doc_entity::Column::ExtractedInformation.is_not_null())
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut nodes_map: HashMap<String, NodeData> = HashMap::new();
    let mut edges: Vec<CytoscapeEdge> = Vec::new();
    let mut edge_counter = 1;

    for doc in &docs {
        if let Some(info) = &doc.extracted_information {
            let doc_title = doc.title.clone();
            extract_evidentiary_elements(
                info,
                &doc_title,
                &mut nodes_map,
                &mut edges,
                &mut edge_counter,
            );
        }
    }

    let evidentiary_count = edges.len();

    let predictions = gnn_link_prediction::Entity::find()
        .all(db)
        .await
        .map_err(DocumentErrors::DatabaseError)?;

    let mut predicted_count = 0;
    for pred in predictions {
        if !nodes_map.contains_key(&pred.target_entity_id) {
            nodes_map.insert(
                pred.target_entity_id.clone(),
                NodeData {
                    id: pred.target_entity_id.clone(),
                    label: format_label(&pred.target_entity_id),
                    r#type: pred.target_entity_type.clone(),
                    badge: Some("Suspect".to_string()),
                    risk_score: None,
                    attributes: serde_json::Map::new(),
                },
            );
        }

        if !nodes_map.contains_key(&pred.candidate_entity_id) {
            nodes_map.insert(
                pred.candidate_entity_id.clone(),
                NodeData {
                    id: pred.candidate_entity_id.clone(),
                    label: format_label(&pred.candidate_entity_id),
                    r#type: pred.candidate_entity_type.clone(),
                    badge: Some("Candidate Associate".to_string()),
                    risk_score: Some(pred.link_probability),
                    attributes: serde_json::Map::new(),
                },
            );
        }

        let color = if pred.link_probability >= 0.70 {
            "#ef4444".to_string()
        } else {
            "#f59e0b".to_string()
        };

        let label = format!(
            "{} ({:.1}%)",
            pred.predicted_edge_type,
            pred.link_probability * 100.0
        );

        edges.push(CytoscapeEdge {
            data: EdgeData {
                id: format!("edge-gnn-pred-{}", pred.id),
                source: pred.target_entity_id,
                target: pred.candidate_entity_id,
                label,
                category: "hypothesis".to_string(),
                is_hypothesis: true,
                style: "dashed".to_string(),
                color,
                probability: Some(pred.link_probability),
                recommendation: pred.recommendation,
                source_document: None,
            },
        });
        predicted_count += 1;
    }

    let nodes: Vec<CytoscapeNode> = nodes_map
        .into_values()
        .map(|data| CytoscapeNode { data })
        .collect();

    let total_nodes = nodes.len();
    let total_edges = edges.len();

    let response = CytoscapeGraphResponse {
        status: "success".to_string(),
        case_id: None,
        meta: GraphMeta {
            total_nodes,
            total_edges,
            evidentiary_edges: evidentiary_count,
            predicted_edges: predicted_count,
            last_analyzed: chrono::Utc::now().to_rfc3339(),
        },
        elements: CytoscapeElements { nodes, edges },
    };

    tracing::info!(
        "Assembled global multi-case graph: {} nodes, {} edges ({} evidentiary, {} predicted)",
        total_nodes, total_edges, evidentiary_count, predicted_count
    );

    Ok((StatusCode::OK, Json(response)))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_label(id: &str) -> String {
    if let Some(stripped) = id.strip_prefix("ENT-PERSON-") {
        stripped.replace('_', " ")
    } else if let Some(stripped) = id.strip_prefix("ENT-PHONE-") {
        stripped.to_string()
    } else if let Some(stripped) = id.strip_prefix("ENT-ACC-") {
        stripped.to_string()
    } else if let Some(stripped) = id.strip_prefix("ENT-OBJ-") {
        stripped.replace('_', " ")
    } else {
        id.to_string()
    }
}

fn extract_evidentiary_elements(
    info: &serde_json::Value,
    doc_title: &str,
    nodes_map: &mut HashMap<String, NodeData>,
    edges: &mut Vec<CytoscapeEdge>,
    edge_counter: &mut usize,
) {
    let mut doc_nodes: Vec<String> = Vec::new();

    if let serde_json::Value::Object(map) = info {
        for (key, val) in map {
            let k_lower = key.to_lowercase();
            if let Some(s) = val.as_str() {
                let s_clean = s.trim();
                if s_clean.is_empty() {
                    continue;
                }

                if k_lower.contains("name") || k_lower.contains("suspect") || k_lower.contains("person") {
                    let id = format!("ENT-PERSON-{}", s_clean.to_uppercase().replace(' ', "_"));
                    nodes_map.entry(id.clone()).or_insert_with(|| NodeData {
                        id: id.clone(),
                        label: s_clean.to_string(),
                        r#type: "person".to_string(),
                        badge: Some("Person of Interest".to_string()),
                        risk_score: None,
                        attributes: serde_json::Map::new(),
                    });
                    doc_nodes.push(id);
                } else if k_lower.contains("phone") || k_lower.contains("mobile") {
                    let id = format!("ENT-PHONE-{}", s_clean.replace(' ', ""));
                    nodes_map.entry(id.clone()).or_insert_with(|| NodeData {
                        id: id.clone(),
                        label: s_clean.to_string(),
                        r#type: "phone".to_string(),
                        badge: Some("Device".to_string()),
                        risk_score: None,
                        attributes: serde_json::Map::new(),
                    });
                    doc_nodes.push(id);
                } else if k_lower.contains("account") || k_lower.contains("bank") {
                    let id = format!("ENT-ACC-{}", s_clean.replace(' ', ""));
                    nodes_map.entry(id.clone()).or_insert_with(|| NodeData {
                        id: id.clone(),
                        label: s_clean.to_string(),
                        r#type: "financial_account".to_string(),
                        badge: Some("Account".to_string()),
                        risk_score: None,
                        attributes: serde_json::Map::new(),
                    });
                    doc_nodes.push(id);
                } else if k_lower.contains("vehicle") || k_lower.contains("object") {
                    let id = format!("ENT-OBJ-{}", s_clean.to_uppercase().replace(' ', "_"));
                    nodes_map.entry(id.clone()).or_insert_with(|| NodeData {
                        id: id.clone(),
                        label: s_clean.to_string(),
                        r#type: "object".to_string(),
                        badge: Some("Asset".to_string()),
                        risk_score: None,
                        attributes: serde_json::Map::new(),
                    });
                    doc_nodes.push(id);
                }
            }
        }
    }

    // Connect evidentiary edges: only link attributes to owner or use explicit extracted relationships
    // Avoid blanket cross-product wiring of every person to every other person in the same document.
    let mut explicit_links_found = false;
    if let serde_json::Value::Object(map) = info {
        if let Some(relationships) = map.get("relationships").and_then(|r| r.as_array()) {
            for rel in relationships {
                if let (Some(s), Some(t), Some(r)) = (
                    rel.get("source").and_then(|v| v.as_str()),
                    rel.get("target").and_then(|v| v.as_str()),
                    rel.get("relationship").and_then(|v| v.as_str()),
                ) {
                    let s_id = format!("ENT-PERSON-{}", s.trim().to_uppercase().replace(' ', "_"));
                    let t_id = format!("ENT-PERSON-{}", t.trim().to_uppercase().replace(' ', "_"));
                    if nodes_map.contains_key(&s_id) && nodes_map.contains_key(&t_id) {
                        edges.push(CytoscapeEdge {
                            data: EdgeData {
                                id: format!("edge-evid-{}", *edge_counter),
                                source: s_id,
                                target: t_id,
                                label: r.to_string(),
                                category: "evidentiary".to_string(),
                                is_hypothesis: false,
                                style: "solid".to_string(),
                                color: "#64748b".to_string(),
                                probability: None,
                                recommendation: None,
                                source_document: Some(doc_title.to_string()),
                            },
                        });
                        *edge_counter += 1;
                        explicit_links_found = true;
                    }
                }
            }
        }
    }

    // Connect Person -> owned Phone, Account, Object within the document
    for i in 0..doc_nodes.len() {
        for j in (i + 1)..doc_nodes.len() {
            let src = &doc_nodes[i];
            let dst = &doc_nodes[j];
            let is_person_attr = (src.contains("PERSON") && (dst.contains("PHONE") || dst.contains("ACC") || dst.contains("OBJ")))
                || (dst.contains("PERSON") && (src.contains("PHONE") || src.contains("ACC") || src.contains("OBJ")));
            
            if is_person_attr {
                let (source, target, label) = if src.contains("PERSON") {
                    let lbl = if dst.contains("PHONE") || dst.contains("ACC") { "owns" } else { "linked_to" };
                    (src.clone(), dst.clone(), lbl)
                } else {
                    let lbl = if src.contains("PHONE") || src.contains("ACC") { "owns" } else { "linked_to" };
                    (dst.clone(), src.clone(), lbl)
                };

                edges.push(CytoscapeEdge {
                    data: EdgeData {
                        id: format!("edge-evid-{}", *edge_counter),
                        source,
                        target,
                        label: label.to_string(),
                        category: "evidentiary".to_string(),
                        is_hypothesis: false,
                        style: "solid".to_string(),
                        color: "#64748b".to_string(),
                        probability: None,
                        recommendation: None,
                        source_document: Some(doc_title.to_string()),
                    },
                });
                *edge_counter += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Router constructor
// ---------------------------------------------------------------------------

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/api/gnn/trigger", post(trigger_gnn_handler))
        .route("/api/gnn/graph", get(get_global_graph_handler))
        .route("/api/cases/{case_id}/graph", get(get_case_graph_handler))
        .with_state(state)
}
