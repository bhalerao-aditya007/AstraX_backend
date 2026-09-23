// libs/document/src/processors/osint.rs
//
// OSINT enrichment processor — calls the Python HF Space OSINT endpoints
// to run ExifTool, phonenumbers, dnstwist, and Sherlock against selectors
// extracted from document metadata. Findings land in
// `document.extracted_information["osint"]`.

use async_trait::async_trait;
use orm::entity::{
    document,
    sea_orm_active_enums::{DocumentStatus, DocumentType},
};
use sea_orm::{ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter};
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::{
    document::{
        base::Document,
        image::Image,
        text::Text,
        video::Video,
    },
    errors::DocumentErrors,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

// ── Finding schema ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsintFinding {
    pub finding_type: String,
    pub label: String,
    #[serde(default)]
    pub attributes: serde_json::Value,
    pub confidence: f64,
    #[serde(default)]
    pub source_url: Option<String>,
    pub tool: String,
    #[serde(default)]
    pub tool_args: serde_json::Value,
    pub raw_sha256: String,
    pub collected_at: String,
    #[serde(default)]
    pub egress_used: bool,
}

#[derive(Debug, Deserialize)]
struct ScanResponse {
    #[serde(default)]
    findings: Vec<OsintFinding>,
    #[allow(dead_code)]
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExifResponse {
    #[serde(default)]
    findings: Vec<OsintFinding>,
}

// ── Processor ───────────────────────────────────────────────────────────

pub struct OsintProcessor {
    base_url: String,
    client: reqwest::Client,
    allow_egress: bool,
}

impl OsintProcessor {
    pub fn from_env() -> Self {
        let base_url = std::env::var("OSINT_SERVICE_URL")
            .or_else(|_| std::env::var("HF_SPACE_URL"))
            .unwrap_or_else(|_| "http://localhost:7860".to_string());

        let allow_egress = std::env::var("OSINT_ALLOW_EGRESS")
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(true);

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(240))
                .build()
                .unwrap_or_default(),
            allow_egress,
        }
    }

    /// Walk arbitrary `extracted_information` JSON and pull OSINT selectors
    /// (phone numbers, usernames, domains) from field names.
    pub fn extract_selectors(val: &serde_json::Value) -> Vec<(String, String)> {
        let mut out = Vec::new();
        Self::walk(val, &mut out);
        out.sort();
        out.dedup();
        out.truncate(12); // demo guard — each selector is a network round trip
        out
    }

    fn walk(val: &serde_json::Value, out: &mut Vec<(String, String)>) {
        match val {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    let kl = k.to_lowercase();
                    if let Some(s) = v.as_str() {
                        let s = s.trim();
                        if s.is_empty() {
                            continue;
                        }
                        let kind = if kl.contains("phone")
                            || kl.contains("mobile")
                            || kl.contains("imei_number")
                            || kl.contains("contact")
                        {
                            Some("phone")
                        } else if kl.contains("username")
                            || kl.contains("handle")
                            || kl.contains("alias")
                            || kl.contains("profile_id")
                        {
                            Some("username")
                        } else if kl.contains("domain")
                            || kl.contains("website")
                            || kl.contains("url")
                        {
                            Some("domain")
                        } else {
                            None
                        };

                        if let Some(kind) = kind {
                            let cleaned = match kind {
                                "phone" => s
                                    .chars()
                                    .filter(|c| c.is_ascii_digit() || *c == '+')
                                    .collect::<String>(),
                                "domain" => s
                                    .replace("https://", "")
                                    .replace("http://", "")
                                    .split('/')
                                    .next()
                                    .unwrap_or("")
                                    .to_string(),
                                _ => s.replace(' ', "_"),
                            };
                            if cleaned.len() >= 3 {
                                out.push((kind.to_string(), cleaned));
                            }
                        }
                    } else {
                        Self::walk(v, out);
                    }
                }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    Self::walk(v, out);
                }
            }
            _ => {}
        }
    }

    /// POST /api/v1/osint/scan — selector-driven enrichment.
    pub async fn scan(&self, selector_type: &str, value: &str) -> Vec<OsintFinding> {
        // Offline-only selectors (phone) still run when egress is disabled
        if !self.allow_egress && selector_type != "phone" {
            return Vec::new();
        }

        let url = format!("{}/api/v1/osint/scan", self.base_url);
        let body = serde_json::json!({
            "selector_type": selector_type,
            "selector_value": value,
        });

        match self.client.post(&url).json(&body).send().await {
            Ok(r) if r.status().is_success() => r
                .json::<ScanResponse>()
                .await
                .map(|x| x.findings)
                .unwrap_or_default(),
            Ok(r) => {
                tracing::warn!(
                    "[OsintProcessor] scan {}={} returned {}",
                    selector_type,
                    value,
                    r.status()
                );
                Vec::new()
            }
            Err(e) => {
                tracing::warn!(
                    "[OsintProcessor] scan {}={} failed: {}",
                    selector_type,
                    value,
                    e
                );
                Vec::new()
            }
        }
    }

    /// POST /api/v1/osint/exif — multipart file upload for EXIF extraction.
    pub async fn exif(&self, bytes: Vec<u8>, filename: &str, mime: &str) -> Vec<OsintFinding> {
        let part = match reqwest::multipart::Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str(mime)
        {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        let form = reqwest::multipart::Form::new().part("file", part);
        let url = format!("{}/api/v1/osint/exif", self.base_url);

        match self.client.post(&url).multipart(form).send().await {
            Ok(r) if r.status().is_success() => r
                .json::<ExifResponse>()
                .await
                .map(|x| x.findings)
                .unwrap_or_default(),
            Ok(r) => {
                tracing::warn!("[OsintProcessor] exif returned {}", r.status());
                Vec::new()
            }
            Err(e) => {
                tracing::warn!("[OsintProcessor] exif failed: {}", e);
                Vec::new()
            }
        }
    }
}

// ── DocumentProcessor impl ──────────────────────────────────────────────

#[async_trait]
impl DocumentProcessor for OsintProcessor {
    async fn process(
        &self,
        doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[OsintProcessor][START] doc_id={}", doc_id);

        // Load current extracted_information from DB
        let model = document::Entity::find_by_id(doc_id)
            .one(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document {} not found", doc_id))
            })?;

        let mut info = model
            .extracted_information
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));

        let mut findings: Vec<OsintFinding> = Vec::new();

        // ExifTool on media — offline, always runs
        let mime = doc.mime_type();
        if mime.starts_with("image/") || mime.starts_with("video/") {
            match doc.fetch().await {
                Ok(bytes) => {
                    let exif_findings = self.exif(bytes, "evidence", mime).await;
                    findings.extend(exif_findings);
                }
                Err(e) => {
                    tracing::warn!(
                        "[OsintProcessor] Failed to fetch bytes for doc_id={}: {}",
                        doc_id,
                        e
                    );
                }
            }
        }

        // Selector-driven enrichment from extracted_information fields
        let selectors = Self::extract_selectors(&info);
        for (kind, value) in selectors {
            let scan_findings = self.scan(&kind, &value).await;
            findings.extend(scan_findings);
        }

        // Write findings into extracted_information["osint"]
        if let Some(map) = info.as_object_mut() {
            map.insert(
                "osint".into(),
                serde_json::to_value(&findings).unwrap_or_default(),
            );
            map.insert(
                "osint_scanned_at".into(),
                serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
            );
        }

        doc.update_extracted_information(info, db).await?;

        tracing::info!(
            "[OsintProcessor][FINISH] doc_id={} findings={}",
            doc_id,
            findings.len()
        );
        Ok(())
    }

    async fn cron_func(
        &self,
        db: &DatabaseConnection,
        store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        let docs = document::Entity::find()
            .filter(document::Column::ExtractedInformation.is_not_null())
            .filter(
                document::Column::Status
                    .is_in([DocumentStatus::Success, DocumentStatus::Finish]),
            )
            .all(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?;

        for model in docs {
            // Skip anything already enriched
            if model
                .extracted_information
                .as_ref()
                .and_then(|i| i.get("osint"))
                .is_some()
            {
                continue;
            }

            let doc_id = model.id;
            let key = model.object_key.clone();

            let res = match model.r#type {
                DocumentType::Image => {
                    let img = Image::new(doc_id, key, store.clone());
                    self.process(&img, db).await
                }
                DocumentType::Video => {
                    let vid = Video::new(doc_id, key, store.clone());
                    self.process(&vid, db).await
                }
                _ => {
                    let txt = Text::new(doc_id, key, store.clone());
                    self.process(&txt, db).await
                }
            };

            if let Err(e) = res {
                tracing::error!(
                    "[OsintProcessor] doc_id={} error: {}",
                    doc_id,
                    e
                );
            }
        }

        Ok(())
    }
}
