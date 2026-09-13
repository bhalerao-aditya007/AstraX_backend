use super::sea_orm_active_enums::{DocumentStatus, DocumentType};
use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Eq, Serialize, Deserialize)]
#[sea_orm(table_name = "document")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub status: DocumentStatus,
    pub r#type: DocumentType,
    pub object_key: String,
    pub extracted_information: Option<Json>,
    pub case_id: Uuid,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::case::Entity",
        from = "Column::CaseId",
        to = "super::case::Column::Id",
        on_update = "NoAction",
        on_delete = "Cascade"
    )]
    Case,
}

impl Related<super::case::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Case.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
