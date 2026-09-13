use std::time::Duration;

use reqwest::multipart;
use serde::{Deserialize, Serialize};

use crate::errors::DocumentErrors;

// ---------------------------------------------------------------------------
// Gradio API DTOs
// ---------------------------------------------------------------------------

/// Represents a file reference returned by Gradio `/upload` and consumed by
/// `/api/<fn_name>` prediction endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradioFileRef {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orig_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Wrapper for `/api/<fn_name>` request bodies.
#[derive(Debug, Serialize)]
pub struct GradioPredictRequest {
    pub data: Vec<serde_json::Value>,
}

/// Wrapper for `/api/<fn_name>` response bodies.
#[derive(Debug, Deserialize)]
pub struct GradioPredictResponse {
    pub data: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// GradioClient
// ---------------------------------------------------------------------------

/// HTTP client for communicating with a Gradio app hosted on HuggingFace Spaces.
///
/// The base URL should point to the Space root, e.g.
/// `https://username-spacename.hf.space` or `http://localhost:7860`.
///
/// # Usage
/// ```ignore
/// let client = GradioClient::from_env();
/// let result = client
///     .predict_with_file("fir_ocr", bytes, "fir.jpg", "image/jpeg", vec![])
///     .await?;
/// ```
#[derive(Clone)]
pub struct GradioClient {
    base_url: String,
    client: reqwest::Client,
}

impl GradioClient {
    /// Create a new client for the given Space URL.
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // 5 min for ML inference
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Construct from the `HF_SPACE_URL` environment variable.
    /// Falls back to `http://localhost:7860` for local development.
    pub fn from_env() -> Self {
        let url = std::env::var("HF_SPACE_URL")
            .unwrap_or_else(|_| "http://localhost:7860".to_string());
        Self::new(url)
    }

    /// Returns the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    // -----------------------------------------------------------------------
    // Wake-up helper
    // -----------------------------------------------------------------------

    /// HF Spaces on the free tier sleep after inactivity. This sends a quick
    /// GET to the root to wake the Space and waits briefly for it to spin up.
    async fn wake_space(&self) {
        let url = format!("{}/", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => {
                tracing::debug!(
                    "[GradioClient] Wake ping returned status={}",
                    resp.status()
                );
            }
            Err(e) => {
                tracing::warn!("[GradioClient] Wake ping failed (Space may be cold): {}", e);
            }
        }
    }

    // -----------------------------------------------------------------------
    // File Upload
    // -----------------------------------------------------------------------

    /// Upload a binary file to the Space's `/upload` endpoint.
    ///
    /// Returns a [`GradioFileRef`] that can be serialized into a prediction
    /// request's `data` array.
    pub async fn upload_file(
        &self,
        file_bytes: Vec<u8>,
        filename: &str,
        mime_type: &str,
    ) -> Result<GradioFileRef, DocumentErrors> {
        let file_size = file_bytes.len() as u64;
        let part = multipart::Part::bytes(file_bytes.clone())
            .file_name(filename.to_string())
            .mime_str(mime_type)
            .map_err(|e| {
                DocumentErrors::StorageError(format!("Failed to build multipart part: {}", e))
            })?;

        let form = multipart::Form::new().part("files", part);

        // Try Gradio 5/6 endpoint (/gradio_api/upload), fallback to legacy (/upload)
        let url_v6 = format!("{}/gradio_api/upload", self.base_url);
        let url_legacy = format!("{}/upload", self.base_url);

        let response = match self.client.post(&url_v6).multipart(form).send().await {
            Ok(resp) if resp.status().is_success() => resp,
            _ => {
                let part_fallback = multipart::Part::bytes(file_bytes)
                    .file_name(filename.to_string())
                    .mime_str(mime_type)
                    .map_err(|e| {
                        DocumentErrors::StorageError(format!("Failed to build multipart part: {}", e))
                    })?;
                let form_fallback = multipart::Form::new().part("files", part_fallback);
                self.client
                    .post(&url_legacy)
                    .multipart(form_fallback)
                    .send()
                    .await
                    .map_err(|e| {
                        DocumentErrors::StorageError(format!("Gradio upload failed: {}", e))
                    })?
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(DocumentErrors::StorageError(format!(
                "Gradio upload returned HTTP {}: {}",
                status, body
            )));
        }

        let paths: Vec<String> = response.json().await.map_err(|e| {
            DocumentErrors::StorageError(format!("Failed to parse Gradio upload response: {}", e))
        })?;

        let path = paths.into_iter().next().ok_or_else(|| {
            DocumentErrors::StorageError("Gradio upload returned empty file paths".to_string())
        })?;

        tracing::info!("[GradioClient] File uploaded successfully: {}", path);

        Ok(GradioFileRef {
            path,
            orig_name: Some(filename.to_string()),
            size: Some(file_size),
            mime_type: Some(mime_type.to_string()),
        })
    }

    // -----------------------------------------------------------------------
    // Prediction (Supports both Gradio 5/6 Queue SSE and Legacy /api/)
    // -----------------------------------------------------------------------

    pub async fn predict(
        &self,
        fn_name: &str,
        data: Vec<serde_json::Value>,
    ) -> Result<serde_json::Value, DocumentErrors> {
        let body = GradioPredictRequest { data };

        // 1. Try Gradio 5/6 protocol: POST /gradio_api/call/{fn_name}
        let v6_url = format!("{}/gradio_api/call/{}", self.base_url, fn_name);
        tracing::info!(
            "[GradioClient] Calling prediction endpoint: {} (Gradio 6)",
            v6_url
        );

        let v6_res = self
            .client
            .post(&v6_url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await;

        if let Ok(resp) = v6_res {
            if resp.status().is_success() {
                // If it returned an event_id, read the SSE stream
                let call_resp: serde_json::Value = resp.json().await.unwrap_or_default();
                if let Some(event_id) = call_resp.get("event_id").and_then(|v| v.as_str()) {
                    let stream_url = format!("{}/{}", v6_url, event_id);
                    let stream_res = self.client.get(&stream_url).send().await.map_err(|e| {
                        DocumentErrors::StorageError(format!("Failed to fetch Gradio stream: {}", e))
                    })?;

                    let sse_text = stream_res.text().await.unwrap_or_default();
                    for line in sse_text.lines() {
                        if line.starts_with("data:") {
                            let json_str = line.trim_start_matches("data:").trim();
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                                if let Some(arr) = val.as_array() {
                                    return Ok(arr.first().cloned().unwrap_or(serde_json::Value::Null));
                                }
                                return Ok(val);
                            }
                        }
                    }
                }
            }
        }

        // 2. Fallback to legacy Gradio /api/{fn_name}
        let url = format!("{}/api/{}", self.base_url, fn_name);
        tracing::info!(
            "[GradioClient] Falling back to legacy prediction endpoint: {}",
            url
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                DocumentErrors::StorageError(format!("Network request to {} failed: {}", url, e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let err_body = response.text().await.unwrap_or_default();
            return Err(DocumentErrors::StorageError(format!(
                "Gradio predict {} returned HTTP {}: {}",
                fn_name, status, err_body
            )));
        }

        let resp: GradioPredictResponse = response.json().await.map_err(|e| {
            DocumentErrors::StorageError(format!(
                "Failed to parse Gradio prediction response from {}: {}",
                fn_name, e
            ))
        })?;

        let result = resp.data.into_iter().next().unwrap_or(serde_json::Value::Null);

        tracing::info!(
            "[GradioClient] Prediction from '{}' completed successfully",
            fn_name
        );
        Ok(result)
    }

    /// Upload a file and call a Gradio prediction endpoint in a single step.
    ///
    /// The file reference is prepended to `extra_args` as the first element
    /// of the `data` array.
    pub async fn predict_with_file(
        &self,
        fn_name: &str,
        file_bytes: Vec<u8>,
        filename: &str,
        mime_type: &str,
        extra_args: Vec<serde_json::Value>,
    ) -> Result<serde_json::Value, DocumentErrors> {
        let file_ref = self.upload_file(file_bytes, filename, mime_type).await?;

        let file_ref_json = serde_json::to_value(&file_ref).map_err(|e| {
            DocumentErrors::StorageError(format!("Failed to serialize GradioFileRef: {}", e))
        })?;

        let mut data = vec![file_ref_json];
        data.extend(extra_args);

        self.predict(fn_name, data).await
    }

    /// Call a JSON-in / JSON-out Gradio endpoint.
    ///
    /// Serializes `input` as the single element in the `data` array, calls
    /// `predict`, and returns the result value.
    pub async fn predict_json(
        &self,
        fn_name: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, DocumentErrors> {
        self.predict(fn_name, vec![input]).await
    }

    /// Parse a Gradio result value as JSON.
    ///
    /// Gradio functions sometimes return a JSON string instead of a direct
    /// object. This helper handles both cases — if the value is already an
    /// object/array it's returned as-is; if it's a string we attempt to parse
    /// it as JSON.
    pub fn parse_result(value: serde_json::Value) -> serde_json::Value {
        match &value {
            serde_json::Value::String(s) => {
                serde_json::from_str(s).unwrap_or(value)
            }
            _ => value,
        }
    }
}
