use std::sync::Arc;
use orm::entity::document;
use sea_orm::{ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};
use serde::{Deserialize, Serialize};

use crate::{
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::{
        gnn::{ConspiracyPredictionRequest, ConspiracyPredictionResponse, GnnProcessor},
        mo_matcher::MoMatcherProcessor,
        summarizer::SummarizerProcessor,
        theory::{TheoryGenerationPayload, TheoryProcessor},
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityLead {
    pub entity_id: String,
    pub display_name: String,
    pub score: f64,
    pub components: LeadScoreComponents,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadScoreComponents {
    pub gnn_probability: f64,
    pub centrality: f64,
    pub mo_similarity: f64,
    pub direct_evidence: f64,
    pub recidivism: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub case_id: String,
    pub total_entities: usize,
    pub total_documents_analyzed: usize,
    pub cross_case_connections: usize,
    pub priority_leads: Vec<PriorityLead>,
    pub theories: Vec<serde_json::Value>,
    pub fact_sheet: serde_json::Value,
    pub gnn_predictions: Vec<serde_json::Value>,
    pub mo_matches: serde_json::Value,
    pub analyzed_at: String,
}

pub struct HistoricalAnalyzer {
    gradio_client: Arc<GradioClient>,
    gnn_service_url: String,
}

impl HistoricalAnalyzer {
    pub fn new(gradio_client: Arc<GradioClient>, gnn_service_url: String) -> Self {
        Self {
            gradio_client,
            gnn_service_url,
        }
    }

    pub async fn analyze_case(
        &self,
        case_id: uuid::Uuid,
        db: &DatabaseConnection,
    ) -> Result<AnalysisReport, DocumentErrors> {
        tracing::info!("[HistoricalAnalyzer] Starting analysis for case_id={}", case_id);

        // 1. Collect Evidence
        let case_docs = document::Entity::find()
            .filter(document::Column::CaseId.eq(case_id))
            .filter(document::Column::ExtractedInformation.is_not_null())
            .all(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?;
            
        let mut case_entities_arr = Vec::new();
        for doc in &case_docs {
            if let Some(info) = &doc.extracted_information {
                case_entities_arr.push(info.clone());
            }
        }
        let case_entities = serde_json::json!(case_entities_arr);

        // 2. Cross-Case Entity Search
        let all_docs = document::Entity::find()
            .filter(document::Column::ExtractedInformation.is_not_null())
            .all(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?;

        // 3. Build Collective Graph
        let (graph, person_ids) = GnnProcessor::build_graph_from_documents(&all_docs);
        let total_entities = person_ids.len();

        // 4. GNN Inference
        let mut gnn_predictions_val = Vec::new();
        let mut gnn_probs: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        
        if person_ids.len() >= 2 {
            let target_entity_id = person_ids[0].clone();
            let candidate_entity_ids = person_ids[1..].to_vec();
            
            let request = ConspiracyPredictionRequest {
                case_id: case_id.to_string(),
                target_entity_id: target_entity_id.clone(),
                target_entity_type: "person".to_string(),
                candidate_entity_ids,
                confidence_threshold: 0.10,
                graph: graph.clone(),
            };

            let client = reqwest::Client::new();
            let endpoint = format!("{}/api/v1/graph/predict-conspiracy", self.gnn_service_url);
            
            if let Ok(resp) = client.post(&endpoint).json(&request).send().await {
                if resp.status().is_success() {
                    if let Ok(pred_res) = resp.json::<ConspiracyPredictionResponse>().await {
                        for pred in pred_res.predicted_conspirators {
                            gnn_probs.insert(pred.entity_id.clone(), pred.link_probability);
                            if let Ok(val) = serde_json::to_value(&pred) {
                                gnn_predictions_val.push(val);
                            }
                        }
                    }
                }
            }
        }

        // 5. MO Matching
        let mo_processor = MoMatcherProcessor::new(self.gradio_client.clone());
        let mo_matches = mo_processor.match_case(case_entities.clone()).await.unwrap_or(serde_json::json!({}));

        // 6. Priority Lead Scoring
        let mut priority_leads = Vec::new();
        let max_connections = std::cmp::max(1, person_ids.len()) as f64;
        
        for pid in &person_ids {
            let p_gnn = gnn_probs.get(pid).copied().unwrap_or(0.0);
            let centrality = 1.0 / max_connections;
            let m_mo = 0.0;
            let e_direct = 0.5;
            let h_recidivism = 0.0;
            
            let score = 0.35 * p_gnn + 0.20 * centrality + 0.15 * m_mo + 0.20 * e_direct + 0.10 * h_recidivism;
            
            priority_leads.push(PriorityLead {
                entity_id: pid.clone(),
                display_name: pid.clone(),
                score,
                components: LeadScoreComponents {
                    gnn_probability: p_gnn,
                    centrality,
                    mo_similarity: m_mo,
                    direct_evidence: e_direct,
                    recidivism: h_recidivism,
                }
            });
        }
        priority_leads.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // 7. Theory Generation
        let theory_processor = TheoryProcessor::new(self.gradio_client.clone());
        let theory_payload = TheoryGenerationPayload {
            case_id: case_id.to_string(),
            fact_sheet: case_entities.clone(),
            graph_data: serde_json::to_value(&graph).unwrap_or(serde_json::json!({})),
            mo_matches: mo_matches.clone(),
            forensic_evidence: serde_json::json!({}),
        };
        let theories_res = theory_processor.generate_theories(theory_payload).await.unwrap_or_default();
        let theories_val = theories_res.into_iter().map(|t| serde_json::to_value(t).unwrap_or(serde_json::json!({}))).collect();

        // 8. Summarization
        let summarizer = SummarizerProcessor::new(self.gradio_client.clone());
        let fact_sheet = summarizer.summarize_case(case_entities).await.unwrap_or(serde_json::json!({}));

        // 9. Return AnalysisReport
        Ok(AnalysisReport {
            case_id: case_id.to_string(),
            total_entities,
            total_documents_analyzed: case_docs.len(),
            cross_case_connections: all_docs.len().saturating_sub(case_docs.len()),
            priority_leads,
            theories: theories_val,
            fact_sheet,
            gnn_predictions: gnn_predictions_val,
            mo_matches,
            analyzed_at: chrono::Utc::now().fixed_offset().to_rfc3339(),
        })
    }
}
