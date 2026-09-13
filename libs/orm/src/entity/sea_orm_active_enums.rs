use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum, Serialize, Deserialize)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "document_status")]
pub enum DocumentStatus {
    #[sea_orm(string_value = "pending")]
    Pending,
    #[sea_orm(string_value = "processing")]
    Processing,
    #[sea_orm(string_value = "success")]
    Success,
    #[sea_orm(string_value = "failed")]
    Failed,
    #[sea_orm(string_value = "finish")]
    Finish,
}

#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum, Serialize, Deserialize)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "document_type")]
pub enum DocumentType {
    #[sea_orm(string_value = "image")]
    Image,
    #[sea_orm(string_value = "text")]
    Text,
    #[sea_orm(string_value = "voice")]
    Voice,
    #[sea_orm(string_value = "video")]
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum, Serialize, Deserialize)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "entity_type")]
pub enum EntityType {
    #[sea_orm(string_value = "person")]
    Person,
    #[sea_orm(string_value = "phone")]
    Phone,
    #[sea_orm(string_value = "vehicle")]
    Vehicle,
    #[sea_orm(string_value = "financial_account")]
    FinancialAccount,
    #[sea_orm(string_value = "object")]
    Object,
    #[sea_orm(string_value = "phantom_entity")]
    PhantomEntity,
}

#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum, Serialize, Deserialize)]
#[sea_orm(rs_type = "String", db_type = "Enum", enum_name = "link_source")]
pub enum LinkSource {
    #[sea_orm(string_value = "evidentiary")]
    Evidentiary,
    #[sea_orm(string_value = "gnn_predicted")]
    GnnPredicted,
    #[sea_orm(string_value = "mo_match")]
    MoMatch,
    #[sea_orm(string_value = "financial_trace")]
    FinancialTrace,
    #[sea_orm(string_value = "manual")]
    Manual,
}
