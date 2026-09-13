use sea_orm_migration::{
    prelude::*,
    sea_query::{extension::postgres::Type, ForeignKeyAction::{Cascade, SetNull}},
};

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260913_000001_entities"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .create_type(
                Type::create()
                    .as_enum(EntityType::Type)
                    .values([
                        EntityType::Person,
                        EntityType::Phone,
                        EntityType::Vehicle,
                        EntityType::FinancialAccount,
                        EntityType::Object,
                        EntityType::PhantomEntity,
                    ])
                    .to_owned(),
            )
            .await?;

        manager
            .create_type(
                Type::create()
                    .as_enum(LinkSource::Type)
                    .values([
                        LinkSource::Evidentiary,
                        LinkSource::GnnPredicted,
                        LinkSource::MoMatch,
                        LinkSource::FinancialTrace,
                        LinkSource::Manual,
                    ])
                    .to_owned(),
            )
            .await?;

        manager
            .get_connection()
            .execute_unprepared("ALTER TYPE document_type ADD VALUE IF NOT EXISTS 'video'")
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Entity::Table)
                    .col(ColumnDef::new(Entity::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Entity::EntityType).custom(EntityType::Type).not_null())
                    .col(ColumnDef::new(Entity::CanonicalId).string_len(256).not_null())
                    .col(ColumnDef::new(Entity::DisplayName).string_len(512).not_null())
                    .col(ColumnDef::new(Entity::RawValue).text().not_null())
                    .col(ColumnDef::new(Entity::Confidence).double().not_null().default(1.0))
                    .col(ColumnDef::new(Entity::SourceDocumentId).uuid().null())
                    .col(ColumnDef::new(Entity::CaseId).uuid().null())
                    .col(ColumnDef::new(Entity::Metadata).json().null())
                    .col(ColumnDef::new(Entity::CreatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .col(ColumnDef::new(Entity::UpdatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_entity_source_document_id")
                            .from(Entity::Table, Entity::SourceDocumentId)
                            .to(Document::Table, Document::Id)
                            .on_delete(SetNull),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_entity_case_id")
                            .from(Entity::Table, Entity::CaseId)
                            .to(Case::Table, Case::Id)
                            .on_delete(Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_entity_canonical_id")
                    .table(Entity::Table)
                    .col(Entity::CanonicalId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_entity_case_id")
                    .table(Entity::Table)
                    .col(Entity::CaseId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_entity_type")
                    .table(Entity::Table)
                    .col(Entity::EntityType)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(EntityLink::Table)
                    .col(ColumnDef::new(EntityLink::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(EntityLink::SourceEntityId).uuid().not_null())
                    .col(ColumnDef::new(EntityLink::TargetEntityId).uuid().not_null())
                    .col(ColumnDef::new(EntityLink::LinkType).string_len(128).not_null())
                    .col(ColumnDef::new(EntityLink::Source).custom(LinkSource::Type).not_null())
                    .col(ColumnDef::new(EntityLink::Probability).double().not_null().default(1.0))
                    .col(ColumnDef::new(EntityLink::EvidenceSummary).text().null())
                    .col(ColumnDef::new(EntityLink::CreatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .col(ColumnDef::new(EntityLink::UpdatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_entity_link_source_entity_id")
                            .from(EntityLink::Table, EntityLink::SourceEntityId)
                            .to(Entity::Table, Entity::Id)
                            .on_delete(Cascade),
                    )
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_entity_link_target_entity_id")
                            .from(EntityLink::Table, EntityLink::TargetEntityId)
                            .to(Entity::Table, Entity::Id)
                            .on_delete(Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_entity_link_source")
                    .table(EntityLink::Table)
                    .col(EntityLink::SourceEntityId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_entity_link_target")
                    .table(EntityLink::Table)
                    .col(EntityLink::TargetEntityId)
                    .to_owned(),
            )
            .await?;

        manager
            .create_table(
                Table::create()
                    .table(Theory::Table)
                    .col(ColumnDef::new(Theory::Id).uuid().not_null().primary_key())
                    .col(ColumnDef::new(Theory::CaseId).uuid().not_null())
                    .col(ColumnDef::new(Theory::TheoryRank).integer().not_null().default(1))
                    .col(ColumnDef::new(Theory::Title).string_len(512).not_null())
                    .col(ColumnDef::new(Theory::ConfidenceScore).double().not_null().default(0.0))
                    .col(ColumnDef::new(Theory::MotiveCategory).string_len(256).null())
                    .col(ColumnDef::new(Theory::Chronology).json().null())
                    .col(ColumnDef::new(Theory::UnresolvedGaps).json().null())
                    .col(ColumnDef::new(Theory::IsCurrent).boolean().not_null().default(true))
                    .col(ColumnDef::new(Theory::CreatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .col(ColumnDef::new(Theory::UpdatedAt).timestamp_with_time_zone().not_null().default(Expr::current_timestamp()))
                    .foreign_key(
                        ForeignKey::create()
                            .name("fk_theory_case_id")
                            .from(Theory::Table, Theory::CaseId)
                            .to(Case::Table, Case::Id)
                            .on_delete(Cascade),
                    )
                    .to_owned(),
            )
            .await?;

        manager
            .create_index(
                Index::create()
                    .name("idx_theory_case_id")
                    .table(Theory::Table)
                    .col(Theory::CaseId)
                    .to_owned(),
            )
            .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .drop_table(Table::drop().table(Theory::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(EntityLink::Table).to_owned())
            .await?;
        manager
            .drop_table(Table::drop().table(Entity::Table).to_owned())
            .await?;
        
        manager
            .drop_type(Type::drop().name(LinkSource::Type).to_owned())
            .await?;
        manager
            .drop_type(Type::drop().name(EntityType::Type).to_owned())
            .await?;

        Ok(())
    }
}

#[derive(DeriveIden)]
pub enum EntityType {
    #[sea_orm(iden = "entity_type")]
    Type,
    #[sea_orm(iden = "person")]
    Person,
    #[sea_orm(iden = "phone")]
    Phone,
    #[sea_orm(iden = "vehicle")]
    Vehicle,
    #[sea_orm(iden = "financial_account")]
    FinancialAccount,
    #[sea_orm(iden = "object")]
    Object,
    #[sea_orm(iden = "phantom_entity")]
    PhantomEntity,
}

#[derive(DeriveIden)]
pub enum LinkSource {
    #[sea_orm(iden = "link_source")]
    Type,
    #[sea_orm(iden = "evidentiary")]
    Evidentiary,
    #[sea_orm(iden = "gnn_predicted")]
    GnnPredicted,
    #[sea_orm(iden = "mo_match")]
    MoMatch,
    #[sea_orm(iden = "financial_trace")]
    FinancialTrace,
    #[sea_orm(iden = "manual")]
    Manual,
}

#[derive(DeriveIden)]
pub enum Entity {
    Table,
    Id,
    EntityType,
    CanonicalId,
    DisplayName,
    RawValue,
    Confidence,
    SourceDocumentId,
    CaseId,
    Metadata,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
pub enum EntityLink {
    Table,
    Id,
    SourceEntityId,
    TargetEntityId,
    LinkType,
    Source,
    Probability,
    EvidenceSummary,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
pub enum Theory {
    Table,
    Id,
    CaseId,
    TheoryRank,
    Title,
    ConfidenceScore,
    MotiveCategory,
    Chronology,
    UnresolvedGaps,
    IsCurrent,
    CreatedAt,
    UpdatedAt,
}

#[derive(DeriveIden)]
pub enum Document {
    Table,
    Id,
}

#[derive(DeriveIden)]
pub enum Case {
    Table,
    Id,
}
