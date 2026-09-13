use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "theory")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub case_id: Uuid,
    pub theory_rank: i32,
    pub title: String,
    #[sea_orm(column_type = "Double")]
    pub confidence_score: f64,
    pub motive_category: Option<String>,
    pub chronology: Option<Json>,
    pub unresolved_gaps: Option<Json>,
    pub is_current: bool,
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
