-- =============================================================================
-- merge_bib_items_into_file_metadata.sql
--
-- 将独立的书目表 bib_items 合并入文件元数据表 file_metadata
-- （BibItem 模型已合并入 FileMetadata 模型，见 jindai/jindai/models.py）。
--
-- 步骤概览：
--   1) ALTER TABLE 为 file_metadata 增加全部书目列（幂等，IF NOT EXISTS）
--   2) 创建书目索引（title / authors / url / item_type / tags GIN / catalog / doi 唯一）
--   3a) 数据迁移：通过 file_attachments 匹配到的文件行，直接回填书目字段
--   3b) 数据迁移：未匹配任何文件的条目，作为新行插入 file_metadata
--       （path 取第一个附件路径；无附件或路径冲突时使用合成键 bib:<uuid>）
--   4) 验证：确认没有遗漏未迁移的条目
--   5) （验证通过后手动执行）DROP TABLE bib_items
--
-- 兼容性说明：
--   - 原 BibItem.extra (JSONB) 不再单独建列，统一合并入 file_metadata.extdata。
--   - 原 bib_items.title 为 NOT NULL；合并后放宽为可空（普通文件无书目标题）。
--   - date/date_added 延续库中实际的 timestamptz 类型。
--   - file_metadata.path 原有 UNIQUE 约束保留；纯书目条目使用合成 path。
--
-- 可重复执行（幂等）；建议先在事务中演练（BEGIN ... ROLLBACK）。
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 1) 添加书目列（PG11+ 带 DEFAULT 的 ADD COLUMN 为元数据操作，不重写表）
-- -----------------------------------------------------------------------------
ALTER TABLE file_metadata
    ADD COLUMN IF NOT EXISTS item_type        varchar(64),
    ADD COLUMN IF NOT EXISTS title            text,
    ADD COLUMN IF NOT EXISTS authors          text[],
    ADD COLUMN IF NOT EXISTS abstract_note    text,
    ADD COLUMN IF NOT EXISTS publication      varchar(512),
    ADD COLUMN IF NOT EXISTS "date"           timestamptz,
    ADD COLUMN IF NOT EXISTS date_added       timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS volume           varchar(32),
    ADD COLUMN IF NOT EXISTS issue            varchar(32),
    ADD COLUMN IF NOT EXISTS pages            varchar(64),
    ADD COLUMN IF NOT EXISTS doi              varchar(256),
    ADD COLUMN IF NOT EXISTS url              varchar(1024),
    ADD COLUMN IF NOT EXISTS isbn             varchar(32),
    ADD COLUMN IF NOT EXISTS issn             varchar(16),
    ADD COLUMN IF NOT EXISTS archive          varchar(256),
    ADD COLUMN IF NOT EXISTS archive_location varchar(512),
    ADD COLUMN IF NOT EXISTS library_catalog  varchar(256),
    ADD COLUMN IF NOT EXISTS call_number      varchar(128),
    ADD COLUMN IF NOT EXISTS language         varchar(32) DEFAULT 'zh',
    ADD COLUMN IF NOT EXISTS short_title      varchar(256),
    ADD COLUMN IF NOT EXISTS series           varchar(256),
    ADD COLUMN IF NOT EXISTS series_title     varchar(256),
    ADD COLUMN IF NOT EXISTS publisher        varchar(256),
    ADD COLUMN IF NOT EXISTS place            varchar(256),
    ADD COLUMN IF NOT EXISTS cover            varchar(1024) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS notes            text,
    ADD COLUMN IF NOT EXISTS tags             text[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS related          text,
    ADD COLUMN IF NOT EXISTS file_attachments text[] NOT NULL DEFAULT '{}';

-- -----------------------------------------------------------------------------
-- 2) 创建索引（命名与旧 idx_bibitem_* 区分，索引名在库内全局唯一）
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_file_metadata_title     ON file_metadata (title);
CREATE INDEX IF NOT EXISTS idx_file_metadata_authors   ON file_metadata (authors);
CREATE INDEX IF NOT EXISTS idx_file_metadata_url       ON file_metadata (url);
CREATE INDEX IF NOT EXISTS idx_file_metadata_item_type ON file_metadata (item_type);
CREATE INDEX IF NOT EXISTS idx_file_metadata_tags      ON file_metadata USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_file_metadata_catalog   ON file_metadata (library_catalog, call_number);

-- 原 BibItem.doi unique=True：用唯一索引实现（允许任意多行 doi IS NULL，
-- 迁移前已确认 bib_items.doi 无重复值）。
CREATE UNIQUE INDEX IF NOT EXISTS uq_file_metadata_doi ON file_metadata (doi);

-- -----------------------------------------------------------------------------
-- 3a) 回填：file_metadata.path 命中任一 file_attachment 的文件行
--     同一文件被多个条目命中时取 date_added 最新的一条（DISTINCT ON）
-- -----------------------------------------------------------------------------
WITH matches AS (
    SELECT DISTINCT ON (fm.path)
        fm.path AS matched_path,
        b.*
    FROM bib_items b
    JOIN file_metadata fm ON fm.path = ANY (b.file_attachments)
    ORDER BY fm.path, b.date_added DESC NULLS LAST, b.id
)
UPDATE file_metadata fm
SET item_type        = m.item_type,
    title            = m.title,
    authors          = m.authors,
    abstract_note    = m.abstract_note,
    publication      = m.publication,
    "date"           = m."date",
    date_added       = COALESCE(m.date_added, fm.created_at),
    volume           = m.volume,
    issue            = m.issue,
    pages            = m.pages,
    doi              = m.doi,
    url              = m.url,
    isbn             = m.isbn,
    issn             = m.issn,
    archive          = m.archive,
    archive_location = m.archive_location,
    library_catalog  = m.library_catalog,
    call_number      = m.call_number,
    language         = COALESCE(m.language, 'zh'),
    short_title      = m.short_title,
    series           = m.series,
    series_title     = m.series_title,
    publisher        = m.publisher,
    place            = m.place,
    cover            = COALESCE(m.cover, ''),
    notes            = m.notes,
    tags             = COALESCE(m.tags, '{}'),
    related          = m.related,
    file_attachments = COALESCE(m.file_attachments, '{}'),
    extdata          = fm.extdata || COALESCE(m.extra, '{}'::jsonb)
FROM matches m
WHERE fm.path = m.matched_path;

-- -----------------------------------------------------------------------------
-- 3b) 插入：未命中任何文件的条目 → file_metadata 新行
--     path 取第一个附件路径；组内路径冲突（rn > 1）或无附件时使用合成键，
--     保证满足 path 的 UNIQUE 约束。
-- -----------------------------------------------------------------------------
WITH unmatched AS (
    SELECT
        b.*,
        row_number() OVER (
            PARTITION BY COALESCE(b.file_attachments[1], b.id::text)
            ORDER BY b.date_added DESC NULLS LAST, b.id
        ) AS rn
    FROM bib_items b
    WHERE NOT EXISTS (
        SELECT 1 FROM file_metadata fm
        WHERE fm.path = ANY (b.file_attachments)
    )
)
INSERT INTO file_metadata (
    id, path, extension, size_bytes, extdata, created_at, modified_at,
    item_type, title, authors, abstract_note, publication, "date", date_added,
    volume, issue, pages, doi, url, isbn, issn, archive, archive_location,
    library_catalog, call_number, language, short_title, series, series_title,
    publisher, place, cover, notes, tags, related, file_attachments
)
SELECT
    gen_random_uuid(),
    CASE
        WHEN u.rn = 1 THEN COALESCE(u.file_attachments[1], 'bib:' || u.id::text)
        ELSE COALESCE(u.file_attachments[1], 'bib:' || u.id::text) || '#' || u.id::text
    END,
    '',
    0,
    COALESCE(u.extra, '{}'::jsonb),
    COALESCE(u.created_at, now()),
    now(),
    u.item_type,
    u.title,
    u.authors,
    u.abstract_note,
    u.publication,
    u."date",
    COALESCE(u.date_added, now()),
    u.volume,
    u.issue,
    u.pages,
    u.doi,
    u.url,
    u.isbn,
    u.issn,
    u.archive,
    u.archive_location,
    u.library_catalog,
    u.call_number,
    COALESCE(u.language, 'zh'),
    u.short_title,
    u.series,
    u.series_title,
    u.publisher,
    u.place,
    COALESCE(u.cover, ''),
    u.notes,
    COALESCE(u.tags, '{}'),
    u.related,
    COALESCE(u.file_attachments, '{}')
FROM unmatched u
ON CONFLICT (path) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 4) 验证：应返回 0（所有条目均已迁移）
-- -----------------------------------------------------------------------------
SELECT count(*) AS unmigrated_items
FROM bib_items b
WHERE NOT EXISTS (
    SELECT 1 FROM file_metadata fm
    WHERE fm.path = ANY (b.file_attachments)
       OR fm.path = 'bib:' || b.id::text
       OR fm.path = COALESCE(b.file_attachments[1], 'bib:' || b.id::text) || '#' || b.id::text
);

-- 迁移结果抽查
SELECT count(*) AS total_bib_rows,
       count(*) FILTER (WHERE title IS NOT NULL) AS with_title,
       count(*) FILTER (WHERE array_length(file_attachments, 1) > 0) AS with_attachments
FROM file_metadata
WHERE title IS NOT NULL
   OR array_length(file_attachments, 1) > 0;

-- -----------------------------------------------------------------------------
-- 5) 确认验证结果无误后，取消下一行注释以移除旧表
-- -----------------------------------------------------------------------------
-- DROP TABLE IF EXISTS bib_items;

COMMIT;

-- -----------------------------------------------------------------------------
-- 列注释（可选，便于 DBA 理解合并后的表结构）
-- -----------------------------------------------------------------------------
COMMENT ON COLUMN file_metadata.item_type        IS 'Item type (e.g., book, journalArticle, conferencePaper)';
COMMENT ON COLUMN file_metadata.title            IS 'Publication title';
COMMENT ON COLUMN file_metadata.authors          IS 'Author(s) / Creator(s)';
COMMENT ON COLUMN file_metadata.abstract_note    IS 'Abstract or summary';
COMMENT ON COLUMN file_metadata.publication      IS 'Publication name (journal, book title, etc.)';
COMMENT ON COLUMN file_metadata."date"           IS 'Publication date';
COMMENT ON COLUMN file_metadata.date_added       IS 'Added date';
COMMENT ON COLUMN file_metadata.volume           IS 'Volume number';
COMMENT ON COLUMN file_metadata.issue            IS 'Issue number';
COMMENT ON COLUMN file_metadata.pages            IS 'Page range';
COMMENT ON COLUMN file_metadata.doi              IS 'Digital Object Identifier';
COMMENT ON COLUMN file_metadata.url              IS 'URL to publication';
COMMENT ON COLUMN file_metadata.isbn             IS 'ISBN';
COMMENT ON COLUMN file_metadata.issn             IS 'ISSN';
COMMENT ON COLUMN file_metadata.archive          IS 'Archive name (e.g., Zotero, local library)';
COMMENT ON COLUMN file_metadata.archive_location IS 'Location within archive';
COMMENT ON COLUMN file_metadata.library_catalog  IS 'Library catalog name';
COMMENT ON COLUMN file_metadata.call_number      IS 'Call number / shelf location';
COMMENT ON COLUMN file_metadata.language         IS 'Publication language';
COMMENT ON COLUMN file_metadata.short_title      IS 'Short title abbreviation';
COMMENT ON COLUMN file_metadata.series           IS 'Series name';
COMMENT ON COLUMN file_metadata.series_title     IS 'Series title';
COMMENT ON COLUMN file_metadata.publisher        IS 'Publisher name';
COMMENT ON COLUMN file_metadata.place            IS 'Publication place';
COMMENT ON COLUMN file_metadata.cover            IS 'Cover file path';
COMMENT ON COLUMN file_metadata.notes            IS 'User notes';
COMMENT ON COLUMN file_metadata.tags             IS 'Tag list';
COMMENT ON COLUMN file_metadata.related          IS 'Related items/links';
COMMENT ON COLUMN file_metadata.file_attachments IS 'File attachment paths';

