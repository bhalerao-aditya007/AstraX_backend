use std::collections::{HashMap, HashSet};
use async_trait::async_trait;
use orm::entity::{document, gnn_link_prediction};
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, Set,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    document::base::Document,
    errors::DocumentErrors,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

// ---------------------------------------------------------------------------
// GNN Microservice Request & Response DTOs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeCategory {
    pub ids: Vec<String>,
    pub features: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphNodes {
    #[serde(default)]
    pub person: NodeCategory,
    #[serde(default)]
    pub phone: NodeCategory,
    #[serde(default)]
    pub financial_account: NodeCategory,
    #[serde(default)]
    pub object: NodeCategory,
    #[serde(default)]
    pub phantom_entity: NodeCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeRelation {
    pub edge_type: String,
    pub src_indices: Vec<usize>,
    pub dst_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphPayload {
    pub nodes: GraphNodes,
    pub edges: Vec<EdgeRelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConspiracyPredictionRequest {
    pub case_id: String,
    pub target_entity_id: String,
    pub target_entity_type: String,
    pub candidate_entity_ids: Vec<String>,
    pub confidence_threshold: f64,
    pub graph: GraphPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedConspirator {
    pub entity_id: String,
    pub link_probability: f64,
    pub relationship_type: String,
    #[serde(default)]
    pub recommendation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConspiracyPredictionResponse {
    pub status: String,
    pub case_id: Option<String>,
    pub target_entity_id: Option<String>,
    pub confidence_threshold: Option<f64>,
    pub total_predicted: Option<usize>,
    #[serde(default)]
    pub predicted_conspirators: Vec<PredictedConspirator>,
}

// ---------------------------------------------------------------------------
// GnnProcessor Implementation
// ---------------------------------------------------------------------------

/// Heterogeneous Graph Transformer (HGT) GNN document processor.
///
/// Implements `DocumentProcessor` by aggregating extracted entities across
/// documents, synthesizing the multi-relational graph, calling the GNN microservice,
/// and persisting link predictions and statutory recommendations into `gnn_link_prediction`.
pub struct GnnProcessor {
    service_url: String,
    client: reqwest::Client,
}

impl GnnProcessor {
    /// Creates a new `GnnProcessor` pointing to the GNN microservice endpoint.
    pub fn new(service_url: String) -> Self {
        Self {
            service_url: service_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Encode entity text attributes and character n-gram distribution into fixed-dim feature vector.
    fn hash_to_features(text: &str, dim: usize) -> Vec<f64> {
        let mut feats = Vec::with_capacity(dim);
        let len = text.len() as f64;
        let token_count = text.split_whitespace().count() as f64;
        let has_digits = if text.chars().any(|c| c.is_ascii_digit()) { 1.0 } else { 0.0 };
        let has_alias = if text.contains('@') || text.to_uppercase().contains("ALIAS") { 1.0 } else { 0.0 };

        // Dimension 0: Normalized length
        if dim > 0 {
            feats.push((len / 100.0).min(1.0));
        }
        // Dimension 1: Normalized token count
        if dim > 1 {
            feats.push((token_count / 10.0).min(1.0));
        }
        // Dimension 2: Numeric indicator
        if dim > 2 {
            feats.push(has_digits);
        }
        // Dimension 3: Alias marker indicator
        if dim > 3 {
            feats.push(has_alias);
        }

        // Remaining dimensions: Normalized character bucket frequency distribution
        let remaining = dim.saturating_sub(feats.len());
        if remaining > 0 {
            let mut char_counts = vec![0.0f64; remaining];
            for b in text.bytes() {
                let idx = (b as usize) % remaining;
                char_counts[idx] += 1.0;
            }
            let total = len.max(1.0);
            for count in char_counts {
                feats.push((count / total).min(1.0));
            }
        }

        feats
    }

    /// Build a collective GraphPayload from all documents with extracted_information.
    pub fn build_graph_from_documents(
        docs: &[document::Model],
    ) -> (GraphPayload, Vec<String>) {
        let mut person_ids = Vec::new();
        let mut person_features = Vec::new();
        let mut person_map: HashMap<String, usize> = HashMap::new();

        let mut phone_ids = Vec::new();
        let mut phone_features = Vec::new();
        let mut phone_map: HashMap<String, usize> = HashMap::new();

        let mut account_ids = Vec::new();
        let mut account_features = Vec::new();
        let mut account_map: HashMap<String, usize> = HashMap::new();

        let mut object_ids = Vec::new();
        let mut object_features = Vec::new();
        let mut object_map: HashMap<String, usize> = HashMap::new();

        let mut phantom_ids = Vec::new();
        let mut phantom_features = Vec::new();
        let mut phantom_map: HashMap<String, usize> = HashMap::new();

        // Edge candidate buffers: (src_idx, dst_idx)
        let mut calls_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut person_phone_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut person_acc_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut acc_transact_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut person_obj_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut phantom_person_edges: HashSet<(usize, usize)> = HashSet::new();

        for doc in docs {
            let info = match &doc.extracted_information {
                Some(json) => json,
                None => continue,
            };

            // Temporary per-document entity trackers for intra-document relationships
            let mut doc_persons: Vec<usize> = Vec::new();
            let mut doc_phones: Vec<usize> = Vec::new();
            let mut doc_accounts: Vec<usize> = Vec::new();
            let mut doc_objects: Vec<usize> = Vec::new();
            let mut doc_phantoms: Vec<usize> = Vec::new();

            // 1. Check if the JSON has structured pre-formatted entities or graph
            if let Some(nodes) = info.get("nodes").and_then(|n| n.as_object()) {
                if let Some(persons) = nodes.get("person").and_then(|p| p.get("ids")).and_then(|ids| ids.as_array()) {
                    for (idx, p_val) in persons.iter().enumerate() {
                        if let Some(id_str) = p_val.as_str() {
                            let p_idx = *person_map.entry(id_str.to_string()).or_insert_with(|| {
                                person_ids.push(id_str.to_string());
                                let feats = nodes
                                    .get("person")
                                    .and_then(|p| p.get("features"))
                                    .and_then(|f| f.get(idx))
                                    .and_then(|f_arr| f_arr.as_array())
                                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect::<Vec<f64>>())
                                    .unwrap_or_else(|| Self::hash_to_features(id_str, 32));
                                person_features.push(feats);
                                person_ids.len() - 1
                            });
                            doc_persons.push(p_idx);
                        }
                    }
                }
            }

            // 2. Generic key extraction from arbitrary JSON
            Self::extract_from_json_value(
                info,
                &mut person_ids,
                &mut person_features,
                &mut person_map,
                &mut phone_ids,
                &mut phone_features,
                &mut phone_map,
                &mut account_ids,
                &mut account_features,
                &mut account_map,
                &mut object_ids,
                &mut object_features,
                &mut object_map,
                &mut phantom_ids,
                &mut phantom_features,
                &mut phantom_map,
                &mut doc_persons,
                &mut doc_phones,
                &mut doc_accounts,
                &mut doc_objects,
                &mut doc_phantoms,
            );

            // Connect intra-document co-occurrence edges
            for &p in &doc_persons {
                for &ph in &doc_phones {
                    person_phone_edges.insert((p, ph));
                }
                for &acc in &doc_accounts {
                    person_acc_edges.insert((p, acc));
                }
                for &obj in &doc_objects {
                    person_obj_edges.insert((p, obj));
                }
            }

            // Calls between persons in the same document
            for i in 0..doc_persons.len() {
                for j in (i + 1)..doc_persons.len() {
                    calls_edges.insert((doc_persons[i], doc_persons[j]));
                }
            }

            // Transactions between accounts
            for i in 0..doc_accounts.len() {
                for j in (i + 1)..doc_accounts.len() {
                    acc_transact_edges.insert((doc_accounts[i], doc_accounts[j]));
                }
            }

            // Phantom matches
            for &ph in &doc_phantoms {
                for &p in &doc_persons {
                    phantom_person_edges.insert((ph, p));
                }
            }
        }

        // Format edges
        let mut edges = Vec::new();

        if !calls_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = calls_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "person,calls,person".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        if !person_phone_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = person_phone_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "person,owns,phone".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        if !person_acc_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = person_acc_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "person,owns,financial_account".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        if !acc_transact_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = acc_transact_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "financial_account,transacts,financial_account".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        if !person_obj_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = person_obj_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "person,linked_to,object".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        if !phantom_person_edges.is_empty() {
            let (src, dst): (Vec<usize>, Vec<usize>) = phantom_person_edges.into_iter().unzip();
            edges.push(EdgeRelation {
                edge_type: "phantom_entity,partial_match,person".to_string(),
                src_indices: src,
                dst_indices: dst,
            });
        }

        let payload = GraphPayload {
            nodes: GraphNodes {
                person: NodeCategory {
                    ids: person_ids.clone(),
                    features: person_features,
                },
                phone: NodeCategory {
                    ids: phone_ids,
                    features: phone_features,
                },
                financial_account: NodeCategory {
                    ids: account_ids,
                    features: account_features,
                },
                object: NodeCategory {
                    ids: object_ids,
                    features: object_features,
                },
                phantom_entity: NodeCategory {
                    ids: phantom_ids,
                    features: phantom_features,
                },
            },
            edges,
        };

        (payload, person_ids)
    }

    /// Recursively searches arbitrary JSON for investigative entity references
    #[allow(clippy::too_many_arguments)]
    fn extract_from_json_value(
        val: &serde_json::Value,
        person_ids: &mut Vec<String>,
        person_features: &mut Vec<Vec<f64>>,
        person_map: &mut HashMap<String, usize>,
        phone_ids: &mut Vec<String>,
        phone_features: &mut Vec<Vec<f64>>,
        phone_map: &mut HashMap<String, usize>,
        account_ids: &mut Vec<String>,
        account_features: &mut Vec<Vec<f64>>,
        account_map: &mut HashMap<String, usize>,
        object_ids: &mut Vec<String>,
        object_features: &mut Vec<Vec<f64>>,
        object_map: &mut HashMap<String, usize>,
        phantom_ids: &mut Vec<String>,
        phantom_features: &mut Vec<Vec<f64>>,
        phantom_map: &mut HashMap<String, usize>,
        doc_persons: &mut Vec<usize>,
        doc_phones: &mut Vec<usize>,
        doc_accounts: &mut Vec<usize>,
        doc_objects: &mut Vec<usize>,
        doc_phantoms: &mut Vec<usize>,
    ) {
        match val {
            serde_json::Value::Object(map) => {
                for (key, v) in map {
                    let k_lower = key.to_lowercase();
                    if let Some(s) = v.as_str() {
                        let s_clean = s.trim();
                        if s_clean.is_empty() {
                            continue;
                        }

                        if k_lower.contains("phone") || k_lower.contains("mobile") || k_lower.contains("contact") {
                            let id = format!("ENT-PHONE-{}", s_clean.replace(' ', ""));
                            let idx = *phone_map.entry(id.clone()).or_insert_with(|| {
                                phone_ids.push(id.clone());
                                phone_features.push(vec![1.0, 0.85, 0.50]); // 3-dim
                                phone_ids.len() - 1
                            });
                            doc_phones.push(idx);
                        } else if k_lower.contains("name") || k_lower.contains("suspect") || k_lower.contains("accused") || k_lower.contains("person") {
                            let id = format!("ENT-PERSON-{}", s_clean.to_uppercase().replace(' ', "_"));
                            let idx = *person_map.entry(id.clone()).or_insert_with(|| {
                                person_ids.push(id.clone());
                                person_features.push(Self::hash_to_features(s_clean, 32)); // 32-dim
                                person_ids.len() - 1
                            });
                            doc_persons.push(idx);
                        } else if k_lower.contains("account") || k_lower.contains("bank") || k_lower.contains("iban") {
                            let id = format!("ENT-ACC-{}", s_clean.replace(' ', ""));
                            let idx = *account_map.entry(id.clone()).or_insert_with(|| {
                                account_ids.push(id.clone());
                                account_features.push(vec![0.80, 0.75, 0.90, 0.60, 0.70, 0.85, 0.50, 0.65]); // 8-dim
                                account_ids.len() - 1
                            });
                            doc_accounts.push(idx);
                        } else if k_lower.contains("vehicle") || k_lower.contains("car") || k_lower.contains("weapon") || k_lower.contains("object") {
                            let id = format!("ENT-OBJ-{}", s_clean.to_uppercase().replace(' ', "_"));
                            let idx = *object_map.entry(id.clone()).or_insert_with(|| {
                                object_ids.push(id.clone());
                                object_features.push(Self::hash_to_features(s_clean, 17)); // 17-dim
                                object_ids.len() - 1
                            });
                            doc_objects.push(idx);
                        } else if k_lower.contains("phantom") || k_lower.contains("unknown") || k_lower.contains("lead") {
                            let id = format!("ENT-PHANTOM-{}", s_clean.to_uppercase().replace(' ', "_"));
                            let idx = *phantom_map.entry(id.clone()).or_insert_with(|| {
                                phantom_ids.push(id.clone());
                                phantom_features.push(Self::hash_to_features(s_clean, 16)); // 16-dim
                                phantom_ids.len() - 1
                            });
                            doc_phantoms.push(idx);
                        }
                    } else if let Some(arr) = v.as_array() {
                        for item in arr {
                            Self::extract_from_json_value(
                                item,
                                person_ids,
                                person_features,
                                person_map,
                                phone_ids,
                                phone_features,
                                phone_map,
                                account_ids,
                                account_features,
                                account_map,
                                object_ids,
                                object_features,
                                object_map,
                                phantom_ids,
                                phantom_features,
                                phantom_map,
                                doc_persons,
                                doc_phones,
                                doc_accounts,
                                doc_objects,
                                doc_phantoms,
                            );
                        }
                    } else if v.is_object() {
                        Self::extract_from_json_value(
                            v,
                            person_ids,
                            person_features,
                            person_map,
                            phone_ids,
                            phone_features,
                            phone_map,
                            account_ids,
                            account_features,
                            account_map,
                            object_ids,
                            object_features,
                            object_map,
                            phantom_ids,
                            phantom_features,
                            phantom_map,
                            doc_persons,
                            doc_phones,
                            doc_accounts,
                            doc_objects,
                            doc_phantoms,
                        );
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::extract_from_json_value(
                        item,
                        person_ids,
                        person_features,
                        person_map,
                        phone_ids,
                        phone_features,
                        phone_map,
                        account_ids,
                        account_features,
                        account_map,
                        object_ids,
                        object_features,
                        object_map,
                        phantom_ids,
                        phantom_features,
                        phantom_map,
                        doc_persons,
                        doc_phones,
                        doc_accounts,
                        doc_objects,
                        doc_phantoms,
                    );
                }
            }
            _ => {}
        }
    }
}

#[async_trait]
impl DocumentProcessor for GnnProcessor {
    /// Process single document by triggering the collective inference pipeline.
    async fn process(
        &self,
        _doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let dummy_store = S3ObjectStore::new(
            "dummy".to_string(),
            "us-east-1".to_string(),
            "http://localhost:9000",
            "key".to_string(),
            "secret".to_string(),
        );
        self.cron_func(db, &dummy_store).await
    }

    /// Background cron / manual trigger:
    /// 1. Queries all documents from DB where extracted_information IS NOT NULL.
    /// 2. Builds collective multi-case GraphPayload.
    /// 3. Calls GNN microservice: POST /api/v1/graph/predict-conspiracy.
    /// 4. Persists predictions & statutory recommendations into `gnn_link_prediction`.
    async fn cron_func(
        &self,
        db: &DatabaseConnection,
        _store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        tracing::info!("Executing GnnProcessor collective graph inference...");

        // 1. Fetch all documents where extracted_information IS NOT NULL
        let documents = document::Entity::find()
            .filter(document::Column::ExtractedInformation.is_not_null())
            .all(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?;

        if documents.is_empty() {
            tracing::warn!("No documents with extracted_information found. Skipping GNN inference.");
            return Ok(());
        }

        // 2. Synthesize collective GraphPayload across all documents
        let (graph, person_ids) = Self::build_graph_from_documents(&documents);

        if person_ids.len() < 2 {
            tracing::warn!(
                "GNN requires at least 2 person entities for conspiracy prediction (found {}).",
                person_ids.len()
            );
            return Ok(());
        }

        let target_entity_id = person_ids[0].clone();
        let candidate_entity_ids = person_ids[1..].to_vec();
        let confidence_threshold = 0.10;

        let request_payload = ConspiracyPredictionRequest {
            case_id: "collective_investigation".to_string(),
            target_entity_id: target_entity_id.clone(),
            target_entity_type: "person".to_string(),
            candidate_entity_ids: candidate_entity_ids.clone(),
            confidence_threshold,
            graph,
        };

        // 3. POST to GNN microservice
        let endpoint = format!("{}/api/v1/graph/predict-conspiracy", self.service_url);
        tracing::info!("Posting to GNN microservice at {}", endpoint);

        let response = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&request_payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let err_body = response.text().await.unwrap_or_default();
            return Err(DocumentErrors::StorageError(format!(
                "GNN microservice returned {}: {}",
                status, err_body
            )));
        }

        let prediction_res: ConspiracyPredictionResponse = response.json().await.map_err(|e| {
            DocumentErrors::StorageError(format!("Failed to parse GNN response: {}", e))
        })?;

        tracing::info!(
            "GNN inference returned {} predicted conspirators",
            prediction_res.predicted_conspirators.len()
        );

        // 4. Persist predictions into gnn_link_prediction table
        for pred in prediction_res.predicted_conspirators {
            let now = chrono::Utc::now().fixed_offset();
            let active = gnn_link_prediction::ActiveModel {
                id: Set(Uuid::new_v4()),
                target_entity_id: Set(target_entity_id.clone()),
                target_entity_type: Set("person".to_string()),
                candidate_entity_id: Set(pred.entity_id),
                candidate_entity_type: Set("person".to_string()),
                predicted_edge_type: Set(pred.relationship_type),
                link_probability: Set(pred.link_probability),
                is_hypothesis_flagged: Set(pred.link_probability >= confidence_threshold),
                confidence_threshold: Set(Some(confidence_threshold)),
                recommendation: Set(pred.recommendation),
                created_at: Set(now),
                updated_at: Set(now),
            };

            if let Err(e) = active.insert(db).await {
                tracing::error!("Failed to persist gnn_link_prediction: {}", e);
            }
        }

        tracing::info!("Successfully updated gnn_link_prediction records.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orm::entity::sea_orm_active_enums::{DocumentStatus, DocumentType};

    #[test]
    fn test_build_graph_from_documents() {
        let doc1 = document::Model {
            id: Uuid::new_v4(),
            title: "FIR 101".to_string(),
            description: "Police report".to_string(),
            status: DocumentStatus::Finish,
            r#type: DocumentType::Image,
            object_key: "1.jpg".to_string(),
            extracted_information: Some(serde_json::json!({
                "suspect": "Rajesh Sharma",
                "phone": "9871987654",
                "bank_account": "HDFC-501004128912",
                "vehicle": "DL-01-AB-1234"
            })),
            case_id: Uuid::new_v4(),
            created_at: chrono::Utc::now().fixed_offset(),
            updated_at: chrono::Utc::now().fixed_offset(),
        };

        let doc2 = document::Model {
            id: Uuid::new_v4(),
            title: "CDR Log".to_string(),
            description: "Tower dump".to_string(),
            status: DocumentStatus::Finish,
            r#type: DocumentType::Text,
            object_key: "2.txt".to_string(),
            extracted_information: Some(serde_json::json!({
                "accused": "Vikram Malhotra",
                "contact": "9810012345"
            })),
            case_id: Uuid::new_v4(),
            created_at: chrono::Utc::now().fixed_offset(),
            updated_at: chrono::Utc::now().fixed_offset(),
        };

        let (graph, person_ids) = GnnProcessor::build_graph_from_documents(&[doc1, doc2]);

        assert_eq!(person_ids.len(), 2);
        assert_eq!(graph.nodes.person.ids.len(), 2);
        assert_eq!(graph.nodes.person.features[0].len(), 32);
        assert_eq!(graph.nodes.phone.ids.len(), 2);
        assert_eq!(graph.nodes.phone.features[0].len(), 3);
        assert_eq!(graph.nodes.financial_account.ids.len(), 1);
        assert_eq!(graph.nodes.financial_account.features[0].len(), 8);
        assert_eq!(graph.nodes.object.ids.len(), 1);
        assert_eq!(graph.nodes.object.features[0].len(), 17);
    }
}
