use std::time::Duration;

use orm::entity::case::{ActiveModel as CaseActiveModel, Entity as CaseEntity, Model as CaseModel};
use orm::entity::document::{self, ActiveModel, Entity as DocumentEntity, Model as DocumentModel};
use orm::entity::sea_orm_active_enums::DocumentStatus;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, Set,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::errors::DocumentErrors;
use crate::storage::object_store::ObjectStore;
use crate::storage::s3_object_store::S3ObjectStore;

// ---------------------------------------------------------------------------
// DTOs (Data Transfer Objects)
// ---------------------------------------------------------------------------

/// Supported document types for upload and processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentType {
    Image,
    Text,
    Voice,
    Video,
}

impl From<DocumentType> for orm::entity::sea_orm_active_enums::DocumentType {
    fn from(t: DocumentType) -> Self {
        match t {
            DocumentType::Image => orm::entity::sea_orm_active_enums::DocumentType::Image,
            DocumentType::Text => orm::entity::sea_orm_active_enums::DocumentType::Text,
            DocumentType::Voice => orm::entity::sea_orm_active_enums::DocumentType::Voice,
            DocumentType::Video => orm::entity::sea_orm_active_enums::DocumentType::Video,
        }
    }
}

impl From<orm::entity::sea_orm_active_enums::DocumentType> for DocumentType {
    fn from(t: orm::entity::sea_orm_active_enums::DocumentType) -> Self {
        match t {
            orm::entity::sea_orm_active_enums::DocumentType::Image => DocumentType::Image,
            orm::entity::sea_orm_active_enums::DocumentType::Text => DocumentType::Text,
            orm::entity::sea_orm_active_enums::DocumentType::Voice => DocumentType::Voice,
            orm::entity::sea_orm_active_enums::DocumentType::Video => DocumentType::Video,
        }
    }
}

/// Response returned when a new upload is initiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitiateUploadResponse {
    pub document_id: Uuid,
    pub upload_url: String,
    pub object_key: String,
}

/// Serializable Document response DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentResponse {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub status: String,
    pub document_type: String,
    pub object_key: String,
    pub extracted_information: Option<serde_json::Value>,
    pub case_id: Uuid,
    pub created_at: chrono::DateTime<chrono::FixedOffset>,
    pub updated_at: chrono::DateTime<chrono::FixedOffset>,
}

impl From<DocumentModel> for DocumentResponse {
    fn from(m: DocumentModel) -> Self {
        let status = match m.status {
            DocumentStatus::Pending => "pending",
            DocumentStatus::Processing => "processing",
            DocumentStatus::Success => "success",
            DocumentStatus::Failed => "failed",
            DocumentStatus::Finish => "finish",
        };
        let document_type = match DocumentType::from(m.r#type) {
            DocumentType::Image => "image",
            DocumentType::Text => "text",
            DocumentType::Voice => "voice",
            DocumentType::Video => "video",
        };
        Self {
            id: m.id,
            title: m.title,
            description: m.description,
            status: status.to_string(),
            document_type: document_type.to_string(),
            object_key: m.object_key,
            extracted_information: m.extracted_information,
            case_id: m.case_id,
            created_at: m.created_at,
            updated_at: m.updated_at,
        }
    }
}

/// Serializable Case response DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResponse {
    pub id: Uuid,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::FixedOffset>,
    pub updated_at: chrono::DateTime<chrono::FixedOffset>,
}

impl From<CaseModel> for CaseResponse {
    fn from(m: CaseModel) -> Self {
        Self {
            id: m.id,
            name: m.name,
            created_at: m.created_at,
            updated_at: m.updated_at,
        }
    }
}

// ---------------------------------------------------------------------------
// DocumentManager
// ---------------------------------------------------------------------------

/// Orchestrates document lifecycle and case management:
/// upload initiation, confirmation, download URL generation, listing, deletion,
/// and Case CRUD operations.
pub struct DocumentManager {
    db: DatabaseConnection,
    storage: S3ObjectStore,
}

impl DocumentManager {
    /// Create a new `DocumentManager`.
    pub fn new(db: DatabaseConnection, storage: S3ObjectStore) -> Self {
        Self { db, storage }
    }

    /// Returns a reference to the database connection.
    pub fn db(&self) -> &DatabaseConnection {
        &self.db
    }

    /// Returns a reference to the S3 object store.
    pub fn storage(&self) -> &S3ObjectStore {
        &self.storage
    }

    // -----------------------------------------------------------------------
    // Document operations
    // -----------------------------------------------------------------------

    /// Initiate a new document upload.
    ///
    /// 1. Generates a unique S3 object key: `{new_uuid}/{file_name}`
    /// 2. Inserts a database record with status = Pending
    /// 3. Generates a presigned PUT URL (5 min expiry)
    /// 4. Returns the document ID, upload URL, and object key
    pub async fn initiate_upload(
        &self,
        title: String,
        description: String,
        file_name: String,
        case_id: Uuid,
        document_type: impl Into<orm::entity::sea_orm_active_enums::DocumentType>,
    ) -> Result<InitiateUploadResponse, DocumentErrors> {
        let doc_id = Uuid::new_v4();
        let object_key = format!("{}/{}", doc_id, file_name);

        // Insert document record with Pending status
        let now = chrono::Utc::now().fixed_offset();
        let active_model = ActiveModel {
            id: Set(doc_id),
            title: Set(title),
            description: Set(description),
            status: Set(DocumentStatus::Pending),
            r#type: Set(document_type.into()),
            object_key: Set(object_key.clone()),
            extracted_information: Set(None),
            case_id: Set(case_id),
            created_at: Set(now),
            updated_at: Set(now),
        };

        active_model.insert(&self.db).await?;

        // Generate presigned PUT URL (5 minutes)
        let upload_url = self
            .storage
            .presigned_put_url(&object_key, Duration::from_secs(300))
            .await?;

        Ok(InitiateUploadResponse {
            document_id: doc_id,
            upload_url,
            object_key,
        })
    }

    /// Confirm (or fail) a previously initiated upload.
    ///
    /// Updates the document status from Pending to Success or Failed
    /// based on the `success` flag from the frontend.
    pub async fn confirm_upload(
        &self,
        document_id: Uuid,
        success: bool,
    ) -> Result<DocumentResponse, DocumentErrors> {
        let doc = DocumentEntity::find_by_id(document_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document {} not found", document_id))
            })?;

        if doc.status != DocumentStatus::Pending {
            return Err(DocumentErrors::ValidationError(format!(
                "Document {} is not in Pending status (current: {:?})",
                document_id, doc.status
            )));
        }

        let new_status = if success {
            DocumentStatus::Success
        } else {
            DocumentStatus::Failed
        };

        let now = chrono::Utc::now().fixed_offset();
        let mut active_model: ActiveModel = doc.into();
        active_model.status = Set(new_status);
        active_model.updated_at = Set(now);

        let updated = active_model.update(&self.db).await?;
        Ok(DocumentResponse::from(updated))
    }

    /// Generate a presigned download (GET) URL for a document.
    pub async fn get_download_url(
        &self,
        document_id: Uuid,
    ) -> Result<String, DocumentErrors> {
        let doc = DocumentEntity::find_by_id(document_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document {} not found", document_id))
            })?;

        let download_url = self
            .storage
            .presigned_get_url(&doc.object_key, Duration::from_secs(300))
            .await?;

        Ok(download_url)
    }

    /// Get a single document by ID.
    pub async fn get_document(
        &self,
        document_id: Uuid,
    ) -> Result<DocumentResponse, DocumentErrors> {
        let doc = DocumentEntity::find_by_id(document_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document {} not found", document_id))
            })?;
        Ok(DocumentResponse::from(doc))
    }

    /// List documents, optionally filtered by case ID.
    pub async fn list_documents(
        &self,
        case_id: Option<Uuid>,
    ) -> Result<Vec<DocumentResponse>, DocumentErrors> {
        let mut query = DocumentEntity::find();

        if let Some(cid) = case_id {
            query = query.filter(document::Column::CaseId.eq(cid));
        }

        let docs = query.all(&self.db).await?;
        Ok(docs.into_iter().map(DocumentResponse::from).collect())
    }

    /// Delete a document from both S3 and the database.
    pub async fn delete_document(
        &self,
        document_id: Uuid,
    ) -> Result<(), DocumentErrors> {
        let doc = DocumentEntity::find_by_id(document_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document {} not found", document_id))
            })?;

        // Delete from S3 first
        self.storage.delete(&doc.object_key).await?;

        // Then delete from database
        let active_model: ActiveModel = doc.into();
        active_model.delete(&self.db).await?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Case CRUD operations
    // -----------------------------------------------------------------------

    /// Create a new case.
    pub async fn create_case(&self, name: String) -> Result<CaseResponse, DocumentErrors> {
        if name.trim().is_empty() {
            return Err(DocumentErrors::ValidationError(
                "Case name cannot be empty".to_string(),
            ));
        }

        let case_id = Uuid::new_v4();
        let now = chrono::Utc::now().fixed_offset();
        let active_model = CaseActiveModel {
            id: Set(case_id),
            name: Set(name),
            created_at: Set(now),
            updated_at: Set(now),
        };

        let case = active_model.insert(&self.db).await?;
        Ok(CaseResponse::from(case))
    }

    /// Get a single case by ID.
    pub async fn get_case(&self, case_id: Uuid) -> Result<CaseResponse, DocumentErrors> {
        let case = CaseEntity::find_by_id(case_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Case {} not found", case_id))
            })?;
        Ok(CaseResponse::from(case))
    }

    /// List all cases.
    pub async fn list_cases(&self) -> Result<Vec<CaseResponse>, DocumentErrors> {
        let cases = CaseEntity::find().all(&self.db).await?;
        Ok(cases.into_iter().map(CaseResponse::from).collect())
    }

    /// Update a case's name by ID.
    pub async fn update_case(
        &self,
        case_id: Uuid,
        name: String,
    ) -> Result<CaseResponse, DocumentErrors> {
        if name.trim().is_empty() {
            return Err(DocumentErrors::ValidationError(
                "Case name cannot be empty".to_string(),
            ));
        }

        let case = CaseEntity::find_by_id(case_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Case {} not found", case_id))
            })?;

        let now = chrono::Utc::now().fixed_offset();
        let mut active_model: CaseActiveModel = case.into();
        active_model.name = Set(name);
        active_model.updated_at = Set(now);

        let updated = active_model.update(&self.db).await?;
        Ok(CaseResponse::from(updated))
    }

    /// Delete a case by ID.
    /// Foreign key ON DELETE CASCADE in PostgreSQL removes related document records.
    pub async fn delete_case(&self, case_id: Uuid) -> Result<(), DocumentErrors> {
        let case = CaseEntity::find_by_id(case_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Case {} not found", case_id))
            })?;

        let active_model: CaseActiveModel = case.into();
        active_model.delete(&self.db).await?;

        Ok(())
    }
}
