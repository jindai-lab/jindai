--
-- PostgreSQL database dump
--

\restrict cUiQTb26uqSlxRkGrCdcVe6Y7H8gV9CZa5V4qeiQrJuuqKdBBVDG0kN7LcGevtU

-- Dumped from database version 18.0 (Debian 18.0-1.pgdg12+3)
-- Dumped by pg_dump version 18.0 (Debian 18.0-1.pgdg12+3)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: vchord; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vchord WITH SCHEMA public;


--
-- Name: EXTENSION vchord; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION vchord IS 'vchord: Vector database plugin for Postgres, written in Rust, specifically designed for LLM';


--
-- Name: fn_dequeue_on_embedding_insert(); Type: FUNCTION; Schema: public; Owner: myuser
--

CREATE FUNCTION public.fn_dequeue_on_embedding_insert() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- 根据插入到 text_embedding 的复合主键 (id, dataset) 从队列删除
    DELETE FROM embedding_pending_queue 
    WHERE id = NEW.id AND dataset = NEW.dataset;
    
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.fn_dequeue_on_embedding_insert() OWNER TO myuser;

--
-- Name: fn_enqueue_paragraph_embedding(); Type: FUNCTION; Schema: public; Owner: myuser
--

CREATE FUNCTION public.fn_enqueue_paragraph_embedding() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- 只有内容长度大于 10 的才加入队列
    IF LENGTH(NEW.content) > 10 THEN
        INSERT INTO embedding_pending_queue (id, dataset)
        VALUES (NEW.id, NEW.dataset)
        ON CONFLICT DO NOTHING; -- 避免重复
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.fn_enqueue_paragraph_embedding() OWNER TO myuser;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: dataset; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.dataset (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    order_weight integer DEFAULT 0 NOT NULL,
    name character varying(128) NOT NULL,
    tags text[]
);


ALTER TABLE public.dataset OWNER TO myuser;

--
-- Name: TABLE dataset; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON TABLE public.dataset IS '数据集信息表';


--
-- Name: COLUMN dataset.order_weight; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.dataset.order_weight IS '排序权重';


--
-- Name: COLUMN dataset.name; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.dataset.name IS '数据集名称';


--
-- Name: embedding_pending_queue; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.embedding_pending_queue (
    id uuid NOT NULL,
    dataset uuid NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.embedding_pending_queue OWNER TO myuser;

--
-- Name: history; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.history (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    user_id uuid NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    queries jsonb DEFAULT '[]'::jsonb NOT NULL
);


ALTER TABLE public.history OWNER TO myuser;

--
-- Name: TABLE history; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON TABLE public.history IS '用户操作历史表';


--
-- Name: COLUMN history.user_id; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.history.user_id IS '关联用户ID';


--
-- Name: COLUMN history.created_at; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.history.created_at IS '创建时间';


--
-- Name: COLUMN history.queries; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.history.queries IS '查询记录列表（JSON）';


--
-- Name: paragraph; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
)
PARTITION BY HASH (dataset);


ALTER TABLE public.paragraph OWNER TO myuser;

--
-- Name: paragraph_p0; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p0 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p0 OWNER TO myuser;

--
-- Name: paragraph_p1; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p1 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p1 OWNER TO myuser;

--
-- Name: paragraph_p10; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p10 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p10 OWNER TO myuser;

--
-- Name: paragraph_p11; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p11 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p11 OWNER TO myuser;

--
-- Name: paragraph_p12; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p12 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p12 OWNER TO myuser;

--
-- Name: paragraph_p13; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p13 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p13 OWNER TO myuser;

--
-- Name: paragraph_p14; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p14 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p14 OWNER TO myuser;

--
-- Name: paragraph_p15; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p15 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p15 OWNER TO myuser;

--
-- Name: paragraph_p16; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p16 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p16 OWNER TO myuser;

--
-- Name: paragraph_p17; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p17 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p17 OWNER TO myuser;

--
-- Name: paragraph_p18; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p18 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p18 OWNER TO myuser;

--
-- Name: paragraph_p19; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p19 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p19 OWNER TO myuser;

--
-- Name: paragraph_p2; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p2 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p2 OWNER TO myuser;

--
-- Name: paragraph_p20; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p20 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p20 OWNER TO myuser;

--
-- Name: paragraph_p21; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p21 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p21 OWNER TO myuser;

--
-- Name: paragraph_p22; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p22 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p22 OWNER TO myuser;

--
-- Name: paragraph_p23; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p23 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p23 OWNER TO myuser;

--
-- Name: paragraph_p24; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p24 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p24 OWNER TO myuser;

--
-- Name: paragraph_p25; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p25 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p25 OWNER TO myuser;

--
-- Name: paragraph_p26; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p26 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p26 OWNER TO myuser;

--
-- Name: paragraph_p27; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p27 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p27 OWNER TO myuser;

--
-- Name: paragraph_p28; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p28 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p28 OWNER TO myuser;

--
-- Name: paragraph_p29; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p29 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p29 OWNER TO myuser;

--
-- Name: paragraph_p3; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p3 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p3 OWNER TO myuser;

--
-- Name: paragraph_p30; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p30 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p30 OWNER TO myuser;

--
-- Name: paragraph_p31; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p31 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p31 OWNER TO myuser;

--
-- Name: paragraph_p32; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p32 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p32 OWNER TO myuser;

--
-- Name: paragraph_p33; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p33 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p33 OWNER TO myuser;

--
-- Name: paragraph_p34; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p34 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p34 OWNER TO myuser;

--
-- Name: paragraph_p35; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p35 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p35 OWNER TO myuser;

--
-- Name: paragraph_p36; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p36 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p36 OWNER TO myuser;

--
-- Name: paragraph_p37; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p37 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p37 OWNER TO myuser;

--
-- Name: paragraph_p38; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p38 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p38 OWNER TO myuser;

--
-- Name: paragraph_p39; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p39 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p39 OWNER TO myuser;

--
-- Name: paragraph_p4; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p4 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p4 OWNER TO myuser;

--
-- Name: paragraph_p40; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p40 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p40 OWNER TO myuser;

--
-- Name: paragraph_p41; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p41 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p41 OWNER TO myuser;

--
-- Name: paragraph_p42; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p42 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p42 OWNER TO myuser;

--
-- Name: paragraph_p43; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p43 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p43 OWNER TO myuser;

--
-- Name: paragraph_p44; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p44 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p44 OWNER TO myuser;

--
-- Name: paragraph_p45; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p45 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p45 OWNER TO myuser;

--
-- Name: paragraph_p46; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p46 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p46 OWNER TO myuser;

--
-- Name: paragraph_p47; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p47 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p47 OWNER TO myuser;

--
-- Name: paragraph_p48; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p48 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p48 OWNER TO myuser;

--
-- Name: paragraph_p49; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p49 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p49 OWNER TO myuser;

--
-- Name: paragraph_p5; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p5 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p5 OWNER TO myuser;

--
-- Name: paragraph_p50; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p50 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p50 OWNER TO myuser;

--
-- Name: paragraph_p51; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p51 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p51 OWNER TO myuser;

--
-- Name: paragraph_p52; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p52 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p52 OWNER TO myuser;

--
-- Name: paragraph_p53; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p53 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p53 OWNER TO myuser;

--
-- Name: paragraph_p54; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p54 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p54 OWNER TO myuser;

--
-- Name: paragraph_p55; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p55 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p55 OWNER TO myuser;

--
-- Name: paragraph_p56; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p56 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p56 OWNER TO myuser;

--
-- Name: paragraph_p57; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p57 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p57 OWNER TO myuser;

--
-- Name: paragraph_p58; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p58 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p58 OWNER TO myuser;

--
-- Name: paragraph_p59; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p59 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p59 OWNER TO myuser;

--
-- Name: paragraph_p6; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p6 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p6 OWNER TO myuser;

--
-- Name: paragraph_p60; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p60 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p60 OWNER TO myuser;

--
-- Name: paragraph_p61; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p61 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p61 OWNER TO myuser;

--
-- Name: paragraph_p62; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p62 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p62 OWNER TO myuser;

--
-- Name: paragraph_p63; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p63 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p63 OWNER TO myuser;

--
-- Name: paragraph_p7; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p7 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p7 OWNER TO myuser;

--
-- Name: paragraph_p8; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p8 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p8 OWNER TO myuser;

--
-- Name: paragraph_p9; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.paragraph_p9 (
    id uuid DEFAULT public.uuid_generate_v4() CONSTRAINT paragraph_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT paragraph_part_dataset_not_null NOT NULL,
    source_url character varying(1024),
    source_page integer,
    author character varying(128),
    pdate timestamp without time zone,
    outline text,
    content text,
    pagenum text,
    lang character varying(16) DEFAULT 'zh'::character varying,
    extdata jsonb DEFAULT '{}'::jsonb,
    keywords text[]
);


ALTER TABLE public.paragraph_p9 OWNER TO myuser;

--
-- Name: task_dbo; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.task_dbo (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    name character varying(128) NOT NULL,
    params jsonb DEFAULT '{}'::jsonb NOT NULL,
    pipeline jsonb DEFAULT '[]'::jsonb NOT NULL,
    resume_next boolean DEFAULT false,
    last_run timestamp without time zone,
    concurrent integer DEFAULT 3,
    shortcut_map jsonb DEFAULT '{}'::jsonb NOT NULL,
    shared boolean DEFAULT false,
    user_id uuid NOT NULL
);


ALTER TABLE public.task_dbo OWNER TO myuser;

--
-- Name: TABLE task_dbo; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON TABLE public.task_dbo IS '任务表';


--
-- Name: COLUMN task_dbo.name; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.name IS '任务名称';


--
-- Name: COLUMN task_dbo.params; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.params IS '任务参数（JSON）';


--
-- Name: COLUMN task_dbo.pipeline; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.pipeline IS '任务流水线（JSON）';


--
-- Name: COLUMN task_dbo.resume_next; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.resume_next IS '是否允许下次恢复';


--
-- Name: COLUMN task_dbo.last_run; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.last_run IS '最后运行时间';


--
-- Name: COLUMN task_dbo.concurrent; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.concurrent IS '并发数';


--
-- Name: COLUMN task_dbo.shortcut_map; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.shortcut_map IS '快捷方式映射（JSON）';


--
-- Name: COLUMN task_dbo.shared; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.task_dbo.shared IS '是否共享';


--
-- Name: terms; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.terms (
    term text NOT NULL
);


ALTER TABLE public.terms OWNER TO myuser;

--
-- Name: text_embeddings; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
)
PARTITION BY HASH (dataset);


ALTER TABLE public.text_embeddings OWNER TO myuser;

--
-- Name: text_embeddings_p0; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p0 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p0 OWNER TO myuser;

--
-- Name: text_embeddings_p1; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p1 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p1 OWNER TO myuser;

--
-- Name: text_embeddings_p10; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p10 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p10 OWNER TO myuser;

--
-- Name: text_embeddings_p11; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p11 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p11 OWNER TO myuser;

--
-- Name: text_embeddings_p12; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p12 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p12 OWNER TO myuser;

--
-- Name: text_embeddings_p13; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p13 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p13 OWNER TO myuser;

--
-- Name: text_embeddings_p14; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p14 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p14 OWNER TO myuser;

--
-- Name: text_embeddings_p15; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p15 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p15 OWNER TO myuser;

--
-- Name: text_embeddings_p16; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p16 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p16 OWNER TO myuser;

--
-- Name: text_embeddings_p17; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p17 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p17 OWNER TO myuser;

--
-- Name: text_embeddings_p18; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p18 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p18 OWNER TO myuser;

--
-- Name: text_embeddings_p19; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p19 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p19 OWNER TO myuser;

--
-- Name: text_embeddings_p2; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p2 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p2 OWNER TO myuser;

--
-- Name: text_embeddings_p20; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p20 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p20 OWNER TO myuser;

--
-- Name: text_embeddings_p21; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p21 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p21 OWNER TO myuser;

--
-- Name: text_embeddings_p22; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p22 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p22 OWNER TO myuser;

--
-- Name: text_embeddings_p23; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p23 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p23 OWNER TO myuser;

--
-- Name: text_embeddings_p24; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p24 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p24 OWNER TO myuser;

--
-- Name: text_embeddings_p25; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p25 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p25 OWNER TO myuser;

--
-- Name: text_embeddings_p26; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p26 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p26 OWNER TO myuser;

--
-- Name: text_embeddings_p27; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p27 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p27 OWNER TO myuser;

--
-- Name: text_embeddings_p28; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p28 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p28 OWNER TO myuser;

--
-- Name: text_embeddings_p29; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p29 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p29 OWNER TO myuser;

--
-- Name: text_embeddings_p3; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p3 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p3 OWNER TO myuser;

--
-- Name: text_embeddings_p30; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p30 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p30 OWNER TO myuser;

--
-- Name: text_embeddings_p31; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p31 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p31 OWNER TO myuser;

--
-- Name: text_embeddings_p32; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p32 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p32 OWNER TO myuser;

--
-- Name: text_embeddings_p33; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p33 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p33 OWNER TO myuser;

--
-- Name: text_embeddings_p34; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p34 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p34 OWNER TO myuser;

--
-- Name: text_embeddings_p35; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p35 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p35 OWNER TO myuser;

--
-- Name: text_embeddings_p36; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p36 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p36 OWNER TO myuser;

--
-- Name: text_embeddings_p37; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p37 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p37 OWNER TO myuser;

--
-- Name: text_embeddings_p38; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p38 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p38 OWNER TO myuser;

--
-- Name: text_embeddings_p39; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p39 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p39 OWNER TO myuser;

--
-- Name: text_embeddings_p4; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p4 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p4 OWNER TO myuser;

--
-- Name: text_embeddings_p40; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p40 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p40 OWNER TO myuser;

--
-- Name: text_embeddings_p41; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p41 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p41 OWNER TO myuser;

--
-- Name: text_embeddings_p42; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p42 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p42 OWNER TO myuser;

--
-- Name: text_embeddings_p43; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p43 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p43 OWNER TO myuser;

--
-- Name: text_embeddings_p44; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p44 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p44 OWNER TO myuser;

--
-- Name: text_embeddings_p45; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p45 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p45 OWNER TO myuser;

--
-- Name: text_embeddings_p46; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p46 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p46 OWNER TO myuser;

--
-- Name: text_embeddings_p47; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p47 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p47 OWNER TO myuser;

--
-- Name: text_embeddings_p48; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p48 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p48 OWNER TO myuser;

--
-- Name: text_embeddings_p49; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p49 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p49 OWNER TO myuser;

--
-- Name: text_embeddings_p5; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p5 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p5 OWNER TO myuser;

--
-- Name: text_embeddings_p50; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p50 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p50 OWNER TO myuser;

--
-- Name: text_embeddings_p51; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p51 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p51 OWNER TO myuser;

--
-- Name: text_embeddings_p52; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p52 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p52 OWNER TO myuser;

--
-- Name: text_embeddings_p53; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p53 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p53 OWNER TO myuser;

--
-- Name: text_embeddings_p54; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p54 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p54 OWNER TO myuser;

--
-- Name: text_embeddings_p55; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p55 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p55 OWNER TO myuser;

--
-- Name: text_embeddings_p56; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p56 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p56 OWNER TO myuser;

--
-- Name: text_embeddings_p57; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p57 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p57 OWNER TO myuser;

--
-- Name: text_embeddings_p58; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p58 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p58 OWNER TO myuser;

--
-- Name: text_embeddings_p59; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p59 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p59 OWNER TO myuser;

--
-- Name: text_embeddings_p6; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p6 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p6 OWNER TO myuser;

--
-- Name: text_embeddings_p60; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p60 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p60 OWNER TO myuser;

--
-- Name: text_embeddings_p61; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p61 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p61 OWNER TO myuser;

--
-- Name: text_embeddings_p62; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p62 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p62 OWNER TO myuser;

--
-- Name: text_embeddings_p63; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p63 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p63 OWNER TO myuser;

--
-- Name: text_embeddings_p7; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p7 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p7 OWNER TO myuser;

--
-- Name: text_embeddings_p8; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p8 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p8 OWNER TO myuser;

--
-- Name: text_embeddings_p9; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.text_embeddings_p9 (
    id uuid CONSTRAINT text_embeddings_part_id_not_null NOT NULL,
    dataset uuid CONSTRAINT text_embeddings_part_dataset_not_null NOT NULL,
    chunk_id integer DEFAULT 0 CONSTRAINT text_embeddings_part_chunk_id_not_null NOT NULL,
    embedding public.halfvec(1024) CONSTRAINT text_embeddings_part_embedding_not_null NOT NULL
);


ALTER TABLE public.text_embeddings_p9 OWNER TO myuser;

--
-- Name: user_info; Type: TABLE; Schema: public; Owner: myuser
--

CREATE TABLE public.user_info (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    username character varying(64) NOT NULL,
    roles text[] DEFAULT '{}'::text[] NOT NULL,
    datasets uuid[]
);


ALTER TABLE public.user_info OWNER TO myuser;

--
-- Name: TABLE user_info; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON TABLE public.user_info IS '用户表';


--
-- Name: COLUMN user_info.username; Type: COMMENT; Schema: public; Owner: myuser
--

COMMENT ON COLUMN public.user_info.username IS '用户名';


--
-- Name: paragraph_p0; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p0 FOR VALUES WITH (modulus 64, remainder 0);


--
-- Name: paragraph_p1; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p1 FOR VALUES WITH (modulus 64, remainder 1);


--
-- Name: paragraph_p10; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p10 FOR VALUES WITH (modulus 64, remainder 10);


--
-- Name: paragraph_p11; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p11 FOR VALUES WITH (modulus 64, remainder 11);


--
-- Name: paragraph_p12; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p12 FOR VALUES WITH (modulus 64, remainder 12);


--
-- Name: paragraph_p13; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p13 FOR VALUES WITH (modulus 64, remainder 13);


--
-- Name: paragraph_p14; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p14 FOR VALUES WITH (modulus 64, remainder 14);


--
-- Name: paragraph_p15; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p15 FOR VALUES WITH (modulus 64, remainder 15);


--
-- Name: paragraph_p16; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p16 FOR VALUES WITH (modulus 64, remainder 16);


--
-- Name: paragraph_p17; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p17 FOR VALUES WITH (modulus 64, remainder 17);


--
-- Name: paragraph_p18; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p18 FOR VALUES WITH (modulus 64, remainder 18);


--
-- Name: paragraph_p19; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p19 FOR VALUES WITH (modulus 64, remainder 19);


--
-- Name: paragraph_p2; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p2 FOR VALUES WITH (modulus 64, remainder 2);


--
-- Name: paragraph_p20; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p20 FOR VALUES WITH (modulus 64, remainder 20);


--
-- Name: paragraph_p21; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p21 FOR VALUES WITH (modulus 64, remainder 21);


--
-- Name: paragraph_p22; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p22 FOR VALUES WITH (modulus 64, remainder 22);


--
-- Name: paragraph_p23; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p23 FOR VALUES WITH (modulus 64, remainder 23);


--
-- Name: paragraph_p24; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p24 FOR VALUES WITH (modulus 64, remainder 24);


--
-- Name: paragraph_p25; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p25 FOR VALUES WITH (modulus 64, remainder 25);


--
-- Name: paragraph_p26; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p26 FOR VALUES WITH (modulus 64, remainder 26);


--
-- Name: paragraph_p27; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p27 FOR VALUES WITH (modulus 64, remainder 27);


--
-- Name: paragraph_p28; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p28 FOR VALUES WITH (modulus 64, remainder 28);


--
-- Name: paragraph_p29; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p29 FOR VALUES WITH (modulus 64, remainder 29);


--
-- Name: paragraph_p3; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p3 FOR VALUES WITH (modulus 64, remainder 3);


--
-- Name: paragraph_p30; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p30 FOR VALUES WITH (modulus 64, remainder 30);


--
-- Name: paragraph_p31; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p31 FOR VALUES WITH (modulus 64, remainder 31);


--
-- Name: paragraph_p32; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p32 FOR VALUES WITH (modulus 64, remainder 32);


--
-- Name: paragraph_p33; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p33 FOR VALUES WITH (modulus 64, remainder 33);


--
-- Name: paragraph_p34; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p34 FOR VALUES WITH (modulus 64, remainder 34);


--
-- Name: paragraph_p35; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p35 FOR VALUES WITH (modulus 64, remainder 35);


--
-- Name: paragraph_p36; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p36 FOR VALUES WITH (modulus 64, remainder 36);


--
-- Name: paragraph_p37; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p37 FOR VALUES WITH (modulus 64, remainder 37);


--
-- Name: paragraph_p38; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p38 FOR VALUES WITH (modulus 64, remainder 38);


--
-- Name: paragraph_p39; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p39 FOR VALUES WITH (modulus 64, remainder 39);


--
-- Name: paragraph_p4; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p4 FOR VALUES WITH (modulus 64, remainder 4);


--
-- Name: paragraph_p40; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p40 FOR VALUES WITH (modulus 64, remainder 40);


--
-- Name: paragraph_p41; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p41 FOR VALUES WITH (modulus 64, remainder 41);


--
-- Name: paragraph_p42; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p42 FOR VALUES WITH (modulus 64, remainder 42);


--
-- Name: paragraph_p43; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p43 FOR VALUES WITH (modulus 64, remainder 43);


--
-- Name: paragraph_p44; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p44 FOR VALUES WITH (modulus 64, remainder 44);


--
-- Name: paragraph_p45; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p45 FOR VALUES WITH (modulus 64, remainder 45);


--
-- Name: paragraph_p46; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p46 FOR VALUES WITH (modulus 64, remainder 46);


--
-- Name: paragraph_p47; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p47 FOR VALUES WITH (modulus 64, remainder 47);


--
-- Name: paragraph_p48; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p48 FOR VALUES WITH (modulus 64, remainder 48);


--
-- Name: paragraph_p49; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p49 FOR VALUES WITH (modulus 64, remainder 49);


--
-- Name: paragraph_p5; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p5 FOR VALUES WITH (modulus 64, remainder 5);


--
-- Name: paragraph_p50; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p50 FOR VALUES WITH (modulus 64, remainder 50);


--
-- Name: paragraph_p51; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p51 FOR VALUES WITH (modulus 64, remainder 51);


--
-- Name: paragraph_p52; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p52 FOR VALUES WITH (modulus 64, remainder 52);


--
-- Name: paragraph_p53; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p53 FOR VALUES WITH (modulus 64, remainder 53);


--
-- Name: paragraph_p54; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p54 FOR VALUES WITH (modulus 64, remainder 54);


--
-- Name: paragraph_p55; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p55 FOR VALUES WITH (modulus 64, remainder 55);


--
-- Name: paragraph_p56; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p56 FOR VALUES WITH (modulus 64, remainder 56);


--
-- Name: paragraph_p57; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p57 FOR VALUES WITH (modulus 64, remainder 57);


--
-- Name: paragraph_p58; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p58 FOR VALUES WITH (modulus 64, remainder 58);


--
-- Name: paragraph_p59; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p59 FOR VALUES WITH (modulus 64, remainder 59);


--
-- Name: paragraph_p6; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p6 FOR VALUES WITH (modulus 64, remainder 6);


--
-- Name: paragraph_p60; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p60 FOR VALUES WITH (modulus 64, remainder 60);


--
-- Name: paragraph_p61; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p61 FOR VALUES WITH (modulus 64, remainder 61);


--
-- Name: paragraph_p62; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p62 FOR VALUES WITH (modulus 64, remainder 62);


--
-- Name: paragraph_p63; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p63 FOR VALUES WITH (modulus 64, remainder 63);


--
-- Name: paragraph_p7; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p7 FOR VALUES WITH (modulus 64, remainder 7);


--
-- Name: paragraph_p8; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p8 FOR VALUES WITH (modulus 64, remainder 8);


--
-- Name: paragraph_p9; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph ATTACH PARTITION public.paragraph_p9 FOR VALUES WITH (modulus 64, remainder 9);


--
-- Name: text_embeddings_p0; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p0 FOR VALUES WITH (modulus 64, remainder 0);


--
-- Name: text_embeddings_p1; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p1 FOR VALUES WITH (modulus 64, remainder 1);


--
-- Name: text_embeddings_p10; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p10 FOR VALUES WITH (modulus 64, remainder 10);


--
-- Name: text_embeddings_p11; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p11 FOR VALUES WITH (modulus 64, remainder 11);


--
-- Name: text_embeddings_p12; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p12 FOR VALUES WITH (modulus 64, remainder 12);


--
-- Name: text_embeddings_p13; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p13 FOR VALUES WITH (modulus 64, remainder 13);


--
-- Name: text_embeddings_p14; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p14 FOR VALUES WITH (modulus 64, remainder 14);


--
-- Name: text_embeddings_p15; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p15 FOR VALUES WITH (modulus 64, remainder 15);


--
-- Name: text_embeddings_p16; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p16 FOR VALUES WITH (modulus 64, remainder 16);


--
-- Name: text_embeddings_p17; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p17 FOR VALUES WITH (modulus 64, remainder 17);


--
-- Name: text_embeddings_p18; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p18 FOR VALUES WITH (modulus 64, remainder 18);


--
-- Name: text_embeddings_p19; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p19 FOR VALUES WITH (modulus 64, remainder 19);


--
-- Name: text_embeddings_p2; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p2 FOR VALUES WITH (modulus 64, remainder 2);


--
-- Name: text_embeddings_p20; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p20 FOR VALUES WITH (modulus 64, remainder 20);


--
-- Name: text_embeddings_p21; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p21 FOR VALUES WITH (modulus 64, remainder 21);


--
-- Name: text_embeddings_p22; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p22 FOR VALUES WITH (modulus 64, remainder 22);


--
-- Name: text_embeddings_p23; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p23 FOR VALUES WITH (modulus 64, remainder 23);


--
-- Name: text_embeddings_p24; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p24 FOR VALUES WITH (modulus 64, remainder 24);


--
-- Name: text_embeddings_p25; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p25 FOR VALUES WITH (modulus 64, remainder 25);


--
-- Name: text_embeddings_p26; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p26 FOR VALUES WITH (modulus 64, remainder 26);


--
-- Name: text_embeddings_p27; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p27 FOR VALUES WITH (modulus 64, remainder 27);


--
-- Name: text_embeddings_p28; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p28 FOR VALUES WITH (modulus 64, remainder 28);


--
-- Name: text_embeddings_p29; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p29 FOR VALUES WITH (modulus 64, remainder 29);


--
-- Name: text_embeddings_p3; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p3 FOR VALUES WITH (modulus 64, remainder 3);


--
-- Name: text_embeddings_p30; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p30 FOR VALUES WITH (modulus 64, remainder 30);


--
-- Name: text_embeddings_p31; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p31 FOR VALUES WITH (modulus 64, remainder 31);


--
-- Name: text_embeddings_p32; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p32 FOR VALUES WITH (modulus 64, remainder 32);


--
-- Name: text_embeddings_p33; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p33 FOR VALUES WITH (modulus 64, remainder 33);


--
-- Name: text_embeddings_p34; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p34 FOR VALUES WITH (modulus 64, remainder 34);


--
-- Name: text_embeddings_p35; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p35 FOR VALUES WITH (modulus 64, remainder 35);


--
-- Name: text_embeddings_p36; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p36 FOR VALUES WITH (modulus 64, remainder 36);


--
-- Name: text_embeddings_p37; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p37 FOR VALUES WITH (modulus 64, remainder 37);


--
-- Name: text_embeddings_p38; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p38 FOR VALUES WITH (modulus 64, remainder 38);


--
-- Name: text_embeddings_p39; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p39 FOR VALUES WITH (modulus 64, remainder 39);


--
-- Name: text_embeddings_p4; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p4 FOR VALUES WITH (modulus 64, remainder 4);


--
-- Name: text_embeddings_p40; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p40 FOR VALUES WITH (modulus 64, remainder 40);


--
-- Name: text_embeddings_p41; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p41 FOR VALUES WITH (modulus 64, remainder 41);


--
-- Name: text_embeddings_p42; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p42 FOR VALUES WITH (modulus 64, remainder 42);


--
-- Name: text_embeddings_p43; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p43 FOR VALUES WITH (modulus 64, remainder 43);


--
-- Name: text_embeddings_p44; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p44 FOR VALUES WITH (modulus 64, remainder 44);


--
-- Name: text_embeddings_p45; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p45 FOR VALUES WITH (modulus 64, remainder 45);


--
-- Name: text_embeddings_p46; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p46 FOR VALUES WITH (modulus 64, remainder 46);


--
-- Name: text_embeddings_p47; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p47 FOR VALUES WITH (modulus 64, remainder 47);


--
-- Name: text_embeddings_p48; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p48 FOR VALUES WITH (modulus 64, remainder 48);


--
-- Name: text_embeddings_p49; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p49 FOR VALUES WITH (modulus 64, remainder 49);


--
-- Name: text_embeddings_p5; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p5 FOR VALUES WITH (modulus 64, remainder 5);


--
-- Name: text_embeddings_p50; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p50 FOR VALUES WITH (modulus 64, remainder 50);


--
-- Name: text_embeddings_p51; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p51 FOR VALUES WITH (modulus 64, remainder 51);


--
-- Name: text_embeddings_p52; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p52 FOR VALUES WITH (modulus 64, remainder 52);


--
-- Name: text_embeddings_p53; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p53 FOR VALUES WITH (modulus 64, remainder 53);


--
-- Name: text_embeddings_p54; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p54 FOR VALUES WITH (modulus 64, remainder 54);


--
-- Name: text_embeddings_p55; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p55 FOR VALUES WITH (modulus 64, remainder 55);


--
-- Name: text_embeddings_p56; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p56 FOR VALUES WITH (modulus 64, remainder 56);


--
-- Name: text_embeddings_p57; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p57 FOR VALUES WITH (modulus 64, remainder 57);


--
-- Name: text_embeddings_p58; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p58 FOR VALUES WITH (modulus 64, remainder 58);


--
-- Name: text_embeddings_p59; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p59 FOR VALUES WITH (modulus 64, remainder 59);


--
-- Name: text_embeddings_p6; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p6 FOR VALUES WITH (modulus 64, remainder 6);


--
-- Name: text_embeddings_p60; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p60 FOR VALUES WITH (modulus 64, remainder 60);


--
-- Name: text_embeddings_p61; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p61 FOR VALUES WITH (modulus 64, remainder 61);


--
-- Name: text_embeddings_p62; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p62 FOR VALUES WITH (modulus 64, remainder 62);


--
-- Name: text_embeddings_p63; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p63 FOR VALUES WITH (modulus 64, remainder 63);


--
-- Name: text_embeddings_p7; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p7 FOR VALUES WITH (modulus 64, remainder 7);


--
-- Name: text_embeddings_p8; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p8 FOR VALUES WITH (modulus 64, remainder 8);


--
-- Name: text_embeddings_p9; Type: TABLE ATTACH; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings ATTACH PARTITION public.text_embeddings_p9 FOR VALUES WITH (modulus 64, remainder 9);


--
-- Name: dataset dataset_name_key; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.dataset
    ADD CONSTRAINT dataset_name_key UNIQUE (name);


--
-- Name: dataset dataset_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.dataset
    ADD CONSTRAINT dataset_pkey PRIMARY KEY (id);


--
-- Name: embedding_pending_queue embedding_pending_queue_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.embedding_pending_queue
    ADD CONSTRAINT embedding_pending_queue_pkey PRIMARY KEY (id, dataset);


--
-- Name: history history_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.history
    ADD CONSTRAINT history_pkey PRIMARY KEY (id);


--
-- Name: paragraph paragraph_part_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph
    ADD CONSTRAINT paragraph_part_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p0 paragraph_p0_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p0
    ADD CONSTRAINT paragraph_p0_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p10 paragraph_p10_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p10
    ADD CONSTRAINT paragraph_p10_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p11 paragraph_p11_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p11
    ADD CONSTRAINT paragraph_p11_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p12 paragraph_p12_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p12
    ADD CONSTRAINT paragraph_p12_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p13 paragraph_p13_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p13
    ADD CONSTRAINT paragraph_p13_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p14 paragraph_p14_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p14
    ADD CONSTRAINT paragraph_p14_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p15 paragraph_p15_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p15
    ADD CONSTRAINT paragraph_p15_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p16 paragraph_p16_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p16
    ADD CONSTRAINT paragraph_p16_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p17 paragraph_p17_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p17
    ADD CONSTRAINT paragraph_p17_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p18 paragraph_p18_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p18
    ADD CONSTRAINT paragraph_p18_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p19 paragraph_p19_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p19
    ADD CONSTRAINT paragraph_p19_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p1 paragraph_p1_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p1
    ADD CONSTRAINT paragraph_p1_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p20 paragraph_p20_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p20
    ADD CONSTRAINT paragraph_p20_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p21 paragraph_p21_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p21
    ADD CONSTRAINT paragraph_p21_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p22 paragraph_p22_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p22
    ADD CONSTRAINT paragraph_p22_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p23 paragraph_p23_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p23
    ADD CONSTRAINT paragraph_p23_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p24 paragraph_p24_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p24
    ADD CONSTRAINT paragraph_p24_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p25 paragraph_p25_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p25
    ADD CONSTRAINT paragraph_p25_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p26 paragraph_p26_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p26
    ADD CONSTRAINT paragraph_p26_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p27 paragraph_p27_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p27
    ADD CONSTRAINT paragraph_p27_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p28 paragraph_p28_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p28
    ADD CONSTRAINT paragraph_p28_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p29 paragraph_p29_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p29
    ADD CONSTRAINT paragraph_p29_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p2 paragraph_p2_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p2
    ADD CONSTRAINT paragraph_p2_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p30 paragraph_p30_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p30
    ADD CONSTRAINT paragraph_p30_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p31 paragraph_p31_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p31
    ADD CONSTRAINT paragraph_p31_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p32 paragraph_p32_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p32
    ADD CONSTRAINT paragraph_p32_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p33 paragraph_p33_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p33
    ADD CONSTRAINT paragraph_p33_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p34 paragraph_p34_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p34
    ADD CONSTRAINT paragraph_p34_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p35 paragraph_p35_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p35
    ADD CONSTRAINT paragraph_p35_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p36 paragraph_p36_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p36
    ADD CONSTRAINT paragraph_p36_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p37 paragraph_p37_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p37
    ADD CONSTRAINT paragraph_p37_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p38 paragraph_p38_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p38
    ADD CONSTRAINT paragraph_p38_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p39 paragraph_p39_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p39
    ADD CONSTRAINT paragraph_p39_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p3 paragraph_p3_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p3
    ADD CONSTRAINT paragraph_p3_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p40 paragraph_p40_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p40
    ADD CONSTRAINT paragraph_p40_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p41 paragraph_p41_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p41
    ADD CONSTRAINT paragraph_p41_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p42 paragraph_p42_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p42
    ADD CONSTRAINT paragraph_p42_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p43 paragraph_p43_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p43
    ADD CONSTRAINT paragraph_p43_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p44 paragraph_p44_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p44
    ADD CONSTRAINT paragraph_p44_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p45 paragraph_p45_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p45
    ADD CONSTRAINT paragraph_p45_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p46 paragraph_p46_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p46
    ADD CONSTRAINT paragraph_p46_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p47 paragraph_p47_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p47
    ADD CONSTRAINT paragraph_p47_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p48 paragraph_p48_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p48
    ADD CONSTRAINT paragraph_p48_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p49 paragraph_p49_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p49
    ADD CONSTRAINT paragraph_p49_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p4 paragraph_p4_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p4
    ADD CONSTRAINT paragraph_p4_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p50 paragraph_p50_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p50
    ADD CONSTRAINT paragraph_p50_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p51 paragraph_p51_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p51
    ADD CONSTRAINT paragraph_p51_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p52 paragraph_p52_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p52
    ADD CONSTRAINT paragraph_p52_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p53 paragraph_p53_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p53
    ADD CONSTRAINT paragraph_p53_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p54 paragraph_p54_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p54
    ADD CONSTRAINT paragraph_p54_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p55 paragraph_p55_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p55
    ADD CONSTRAINT paragraph_p55_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p56 paragraph_p56_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p56
    ADD CONSTRAINT paragraph_p56_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p57 paragraph_p57_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p57
    ADD CONSTRAINT paragraph_p57_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p58 paragraph_p58_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p58
    ADD CONSTRAINT paragraph_p58_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p59 paragraph_p59_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p59
    ADD CONSTRAINT paragraph_p59_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p5 paragraph_p5_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p5
    ADD CONSTRAINT paragraph_p5_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p60 paragraph_p60_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p60
    ADD CONSTRAINT paragraph_p60_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p61 paragraph_p61_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p61
    ADD CONSTRAINT paragraph_p61_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p62 paragraph_p62_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p62
    ADD CONSTRAINT paragraph_p62_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p63 paragraph_p63_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p63
    ADD CONSTRAINT paragraph_p63_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p6 paragraph_p6_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p6
    ADD CONSTRAINT paragraph_p6_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p7 paragraph_p7_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p7
    ADD CONSTRAINT paragraph_p7_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p8 paragraph_p8_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p8
    ADD CONSTRAINT paragraph_p8_pkey PRIMARY KEY (id, dataset);


--
-- Name: paragraph_p9 paragraph_p9_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.paragraph_p9
    ADD CONSTRAINT paragraph_p9_pkey PRIMARY KEY (id, dataset);


--
-- Name: terms pk_term; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.terms
    ADD CONSTRAINT pk_term PRIMARY KEY (term);


--
-- Name: task_dbo task_dbo_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.task_dbo
    ADD CONSTRAINT task_dbo_pkey PRIMARY KEY (id);


--
-- Name: text_embeddings text_embeddings_part_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings
    ADD CONSTRAINT text_embeddings_part_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p0 text_embeddings_p0_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p0
    ADD CONSTRAINT text_embeddings_p0_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p10 text_embeddings_p10_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p10
    ADD CONSTRAINT text_embeddings_p10_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p11 text_embeddings_p11_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p11
    ADD CONSTRAINT text_embeddings_p11_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p12 text_embeddings_p12_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p12
    ADD CONSTRAINT text_embeddings_p12_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p13 text_embeddings_p13_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p13
    ADD CONSTRAINT text_embeddings_p13_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p14 text_embeddings_p14_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p14
    ADD CONSTRAINT text_embeddings_p14_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p15 text_embeddings_p15_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p15
    ADD CONSTRAINT text_embeddings_p15_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p16 text_embeddings_p16_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p16
    ADD CONSTRAINT text_embeddings_p16_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p17 text_embeddings_p17_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p17
    ADD CONSTRAINT text_embeddings_p17_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p18 text_embeddings_p18_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p18
    ADD CONSTRAINT text_embeddings_p18_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p19 text_embeddings_p19_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p19
    ADD CONSTRAINT text_embeddings_p19_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p1 text_embeddings_p1_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p1
    ADD CONSTRAINT text_embeddings_p1_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p20 text_embeddings_p20_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p20
    ADD CONSTRAINT text_embeddings_p20_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p21 text_embeddings_p21_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p21
    ADD CONSTRAINT text_embeddings_p21_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p22 text_embeddings_p22_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p22
    ADD CONSTRAINT text_embeddings_p22_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p23 text_embeddings_p23_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p23
    ADD CONSTRAINT text_embeddings_p23_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p24 text_embeddings_p24_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p24
    ADD CONSTRAINT text_embeddings_p24_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p25 text_embeddings_p25_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p25
    ADD CONSTRAINT text_embeddings_p25_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p26 text_embeddings_p26_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p26
    ADD CONSTRAINT text_embeddings_p26_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p27 text_embeddings_p27_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p27
    ADD CONSTRAINT text_embeddings_p27_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p28 text_embeddings_p28_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p28
    ADD CONSTRAINT text_embeddings_p28_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p29 text_embeddings_p29_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p29
    ADD CONSTRAINT text_embeddings_p29_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p2 text_embeddings_p2_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p2
    ADD CONSTRAINT text_embeddings_p2_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p30 text_embeddings_p30_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p30
    ADD CONSTRAINT text_embeddings_p30_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p31 text_embeddings_p31_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p31
    ADD CONSTRAINT text_embeddings_p31_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p32 text_embeddings_p32_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p32
    ADD CONSTRAINT text_embeddings_p32_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p33 text_embeddings_p33_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p33
    ADD CONSTRAINT text_embeddings_p33_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p34 text_embeddings_p34_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p34
    ADD CONSTRAINT text_embeddings_p34_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p35 text_embeddings_p35_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p35
    ADD CONSTRAINT text_embeddings_p35_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p36 text_embeddings_p36_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p36
    ADD CONSTRAINT text_embeddings_p36_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p37 text_embeddings_p37_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p37
    ADD CONSTRAINT text_embeddings_p37_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p38 text_embeddings_p38_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p38
    ADD CONSTRAINT text_embeddings_p38_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p39 text_embeddings_p39_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p39
    ADD CONSTRAINT text_embeddings_p39_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p3 text_embeddings_p3_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p3
    ADD CONSTRAINT text_embeddings_p3_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p40 text_embeddings_p40_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p40
    ADD CONSTRAINT text_embeddings_p40_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p41 text_embeddings_p41_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p41
    ADD CONSTRAINT text_embeddings_p41_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p42 text_embeddings_p42_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p42
    ADD CONSTRAINT text_embeddings_p42_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p43 text_embeddings_p43_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p43
    ADD CONSTRAINT text_embeddings_p43_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p44 text_embeddings_p44_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p44
    ADD CONSTRAINT text_embeddings_p44_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p45 text_embeddings_p45_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p45
    ADD CONSTRAINT text_embeddings_p45_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p46 text_embeddings_p46_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p46
    ADD CONSTRAINT text_embeddings_p46_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p47 text_embeddings_p47_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p47
    ADD CONSTRAINT text_embeddings_p47_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p48 text_embeddings_p48_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p48
    ADD CONSTRAINT text_embeddings_p48_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p49 text_embeddings_p49_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p49
    ADD CONSTRAINT text_embeddings_p49_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p4 text_embeddings_p4_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p4
    ADD CONSTRAINT text_embeddings_p4_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p50 text_embeddings_p50_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p50
    ADD CONSTRAINT text_embeddings_p50_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p51 text_embeddings_p51_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p51
    ADD CONSTRAINT text_embeddings_p51_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p52 text_embeddings_p52_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p52
    ADD CONSTRAINT text_embeddings_p52_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p53 text_embeddings_p53_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p53
    ADD CONSTRAINT text_embeddings_p53_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p54 text_embeddings_p54_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p54
    ADD CONSTRAINT text_embeddings_p54_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p55 text_embeddings_p55_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p55
    ADD CONSTRAINT text_embeddings_p55_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p56 text_embeddings_p56_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p56
    ADD CONSTRAINT text_embeddings_p56_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p57 text_embeddings_p57_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p57
    ADD CONSTRAINT text_embeddings_p57_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p58 text_embeddings_p58_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p58
    ADD CONSTRAINT text_embeddings_p58_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p59 text_embeddings_p59_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p59
    ADD CONSTRAINT text_embeddings_p59_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p5 text_embeddings_p5_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p5
    ADD CONSTRAINT text_embeddings_p5_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p60 text_embeddings_p60_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p60
    ADD CONSTRAINT text_embeddings_p60_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p61 text_embeddings_p61_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p61
    ADD CONSTRAINT text_embeddings_p61_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p62 text_embeddings_p62_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p62
    ADD CONSTRAINT text_embeddings_p62_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p63 text_embeddings_p63_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p63
    ADD CONSTRAINT text_embeddings_p63_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p6 text_embeddings_p6_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p6
    ADD CONSTRAINT text_embeddings_p6_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p7 text_embeddings_p7_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p7
    ADD CONSTRAINT text_embeddings_p7_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p8 text_embeddings_p8_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p8
    ADD CONSTRAINT text_embeddings_p8_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: text_embeddings_p9 text_embeddings_p9_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.text_embeddings_p9
    ADD CONSTRAINT text_embeddings_p9_pkey PRIMARY KEY (id, chunk_id, dataset);


--
-- Name: user_info user_info_pkey; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.user_info
    ADD CONSTRAINT user_info_pkey PRIMARY KEY (id);


--
-- Name: user_info user_info_username_key; Type: CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.user_info
    ADD CONSTRAINT user_info_username_key UNIQUE (username);


--
-- Name: fki_dataset; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX fki_dataset ON ONLY public.paragraph USING btree (dataset);


--
-- Name: fki_fk_user_id; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX fki_fk_user_id ON public.task_dbo USING btree (user_id);


--
-- Name: idx_paragraph_author; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_author ON ONLY public.paragraph USING btree (author);


--
-- Name: idx_paragraph_keywords; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_keywords ON ONLY public.paragraph USING gin (keywords);


--
-- Name: idx_paragraph_outline; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_outline ON ONLY public.paragraph USING btree (outline);


--
-- Name: idx_paragraph_pagenum; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_pagenum ON ONLY public.paragraph USING btree (pagenum);


--
-- Name: idx_paragraph_pdate; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_pdate ON ONLY public.paragraph USING btree (pdate);


--
-- Name: idx_paragraph_source; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_paragraph_source ON ONLY public.paragraph USING btree (source_url, source_page, pagenum);


--
-- Name: idx_text_embeddings_vchord; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX idx_text_embeddings_vchord ON ONLY public.text_embeddings USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: paragraph_p0_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_author_idx ON public.paragraph_p0 USING btree (author);


--
-- Name: paragraph_p0_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_dataset_idx ON public.paragraph_p0 USING btree (dataset);


--
-- Name: paragraph_p0_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_keywords_idx ON public.paragraph_p0 USING gin (keywords);


--
-- Name: paragraph_p0_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_outline_idx ON public.paragraph_p0 USING btree (outline);


--
-- Name: paragraph_p0_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_pagenum_idx ON public.paragraph_p0 USING btree (pagenum);


--
-- Name: paragraph_p0_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_pdate_idx ON public.paragraph_p0 USING btree (pdate);


--
-- Name: paragraph_p0_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p0_source_url_source_page_pagenum_idx ON public.paragraph_p0 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p10_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_author_idx ON public.paragraph_p10 USING btree (author);


--
-- Name: paragraph_p10_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_dataset_idx ON public.paragraph_p10 USING btree (dataset);


--
-- Name: paragraph_p10_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_keywords_idx ON public.paragraph_p10 USING gin (keywords);


--
-- Name: paragraph_p10_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_outline_idx ON public.paragraph_p10 USING btree (outline);


--
-- Name: paragraph_p10_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_pagenum_idx ON public.paragraph_p10 USING btree (pagenum);


--
-- Name: paragraph_p10_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_pdate_idx ON public.paragraph_p10 USING btree (pdate);


--
-- Name: paragraph_p10_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p10_source_url_source_page_pagenum_idx ON public.paragraph_p10 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p11_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_author_idx ON public.paragraph_p11 USING btree (author);


--
-- Name: paragraph_p11_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_dataset_idx ON public.paragraph_p11 USING btree (dataset);


--
-- Name: paragraph_p11_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_keywords_idx ON public.paragraph_p11 USING gin (keywords);


--
-- Name: paragraph_p11_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_outline_idx ON public.paragraph_p11 USING btree (outline);


--
-- Name: paragraph_p11_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_pagenum_idx ON public.paragraph_p11 USING btree (pagenum);


--
-- Name: paragraph_p11_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_pdate_idx ON public.paragraph_p11 USING btree (pdate);


--
-- Name: paragraph_p11_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p11_source_url_source_page_pagenum_idx ON public.paragraph_p11 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p12_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_author_idx ON public.paragraph_p12 USING btree (author);


--
-- Name: paragraph_p12_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_dataset_idx ON public.paragraph_p12 USING btree (dataset);


--
-- Name: paragraph_p12_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_keywords_idx ON public.paragraph_p12 USING gin (keywords);


--
-- Name: paragraph_p12_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_outline_idx ON public.paragraph_p12 USING btree (outline);


--
-- Name: paragraph_p12_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_pagenum_idx ON public.paragraph_p12 USING btree (pagenum);


--
-- Name: paragraph_p12_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_pdate_idx ON public.paragraph_p12 USING btree (pdate);


--
-- Name: paragraph_p12_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p12_source_url_source_page_pagenum_idx ON public.paragraph_p12 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p13_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_author_idx ON public.paragraph_p13 USING btree (author);


--
-- Name: paragraph_p13_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_dataset_idx ON public.paragraph_p13 USING btree (dataset);


--
-- Name: paragraph_p13_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_keywords_idx ON public.paragraph_p13 USING gin (keywords);


--
-- Name: paragraph_p13_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_outline_idx ON public.paragraph_p13 USING btree (outline);


--
-- Name: paragraph_p13_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_pagenum_idx ON public.paragraph_p13 USING btree (pagenum);


--
-- Name: paragraph_p13_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_pdate_idx ON public.paragraph_p13 USING btree (pdate);


--
-- Name: paragraph_p13_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p13_source_url_source_page_pagenum_idx ON public.paragraph_p13 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p14_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_author_idx ON public.paragraph_p14 USING btree (author);


--
-- Name: paragraph_p14_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_dataset_idx ON public.paragraph_p14 USING btree (dataset);


--
-- Name: paragraph_p14_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_keywords_idx ON public.paragraph_p14 USING gin (keywords);


--
-- Name: paragraph_p14_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_outline_idx ON public.paragraph_p14 USING btree (outline);


--
-- Name: paragraph_p14_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_pagenum_idx ON public.paragraph_p14 USING btree (pagenum);


--
-- Name: paragraph_p14_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_pdate_idx ON public.paragraph_p14 USING btree (pdate);


--
-- Name: paragraph_p14_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p14_source_url_source_page_pagenum_idx ON public.paragraph_p14 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p15_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_author_idx ON public.paragraph_p15 USING btree (author);


--
-- Name: paragraph_p15_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_dataset_idx ON public.paragraph_p15 USING btree (dataset);


--
-- Name: paragraph_p15_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_keywords_idx ON public.paragraph_p15 USING gin (keywords);


--
-- Name: paragraph_p15_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_outline_idx ON public.paragraph_p15 USING btree (outline);


--
-- Name: paragraph_p15_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_pagenum_idx ON public.paragraph_p15 USING btree (pagenum);


--
-- Name: paragraph_p15_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_pdate_idx ON public.paragraph_p15 USING btree (pdate);


--
-- Name: paragraph_p15_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p15_source_url_source_page_pagenum_idx ON public.paragraph_p15 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p16_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_author_idx ON public.paragraph_p16 USING btree (author);


--
-- Name: paragraph_p16_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_dataset_idx ON public.paragraph_p16 USING btree (dataset);


--
-- Name: paragraph_p16_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_keywords_idx ON public.paragraph_p16 USING gin (keywords);


--
-- Name: paragraph_p16_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_outline_idx ON public.paragraph_p16 USING btree (outline);


--
-- Name: paragraph_p16_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_pagenum_idx ON public.paragraph_p16 USING btree (pagenum);


--
-- Name: paragraph_p16_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_pdate_idx ON public.paragraph_p16 USING btree (pdate);


--
-- Name: paragraph_p16_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p16_source_url_source_page_pagenum_idx ON public.paragraph_p16 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p17_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_author_idx ON public.paragraph_p17 USING btree (author);


--
-- Name: paragraph_p17_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_dataset_idx ON public.paragraph_p17 USING btree (dataset);


--
-- Name: paragraph_p17_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_keywords_idx ON public.paragraph_p17 USING gin (keywords);


--
-- Name: paragraph_p17_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_outline_idx ON public.paragraph_p17 USING btree (outline);


--
-- Name: paragraph_p17_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_pagenum_idx ON public.paragraph_p17 USING btree (pagenum);


--
-- Name: paragraph_p17_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_pdate_idx ON public.paragraph_p17 USING btree (pdate);


--
-- Name: paragraph_p17_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p17_source_url_source_page_pagenum_idx ON public.paragraph_p17 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p18_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_author_idx ON public.paragraph_p18 USING btree (author);


--
-- Name: paragraph_p18_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_dataset_idx ON public.paragraph_p18 USING btree (dataset);


--
-- Name: paragraph_p18_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_keywords_idx ON public.paragraph_p18 USING gin (keywords);


--
-- Name: paragraph_p18_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_outline_idx ON public.paragraph_p18 USING btree (outline);


--
-- Name: paragraph_p18_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_pagenum_idx ON public.paragraph_p18 USING btree (pagenum);


--
-- Name: paragraph_p18_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_pdate_idx ON public.paragraph_p18 USING btree (pdate);


--
-- Name: paragraph_p18_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p18_source_url_source_page_pagenum_idx ON public.paragraph_p18 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p19_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_author_idx ON public.paragraph_p19 USING btree (author);


--
-- Name: paragraph_p19_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_dataset_idx ON public.paragraph_p19 USING btree (dataset);


--
-- Name: paragraph_p19_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_keywords_idx ON public.paragraph_p19 USING gin (keywords);


--
-- Name: paragraph_p19_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_outline_idx ON public.paragraph_p19 USING btree (outline);


--
-- Name: paragraph_p19_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_pagenum_idx ON public.paragraph_p19 USING btree (pagenum);


--
-- Name: paragraph_p19_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_pdate_idx ON public.paragraph_p19 USING btree (pdate);


--
-- Name: paragraph_p19_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p19_source_url_source_page_pagenum_idx ON public.paragraph_p19 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p1_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_author_idx ON public.paragraph_p1 USING btree (author);


--
-- Name: paragraph_p1_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_dataset_idx ON public.paragraph_p1 USING btree (dataset);


--
-- Name: paragraph_p1_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_keywords_idx ON public.paragraph_p1 USING gin (keywords);


--
-- Name: paragraph_p1_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_outline_idx ON public.paragraph_p1 USING btree (outline);


--
-- Name: paragraph_p1_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_pagenum_idx ON public.paragraph_p1 USING btree (pagenum);


--
-- Name: paragraph_p1_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_pdate_idx ON public.paragraph_p1 USING btree (pdate);


--
-- Name: paragraph_p1_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p1_source_url_source_page_pagenum_idx ON public.paragraph_p1 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p20_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_author_idx ON public.paragraph_p20 USING btree (author);


--
-- Name: paragraph_p20_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_dataset_idx ON public.paragraph_p20 USING btree (dataset);


--
-- Name: paragraph_p20_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_keywords_idx ON public.paragraph_p20 USING gin (keywords);


--
-- Name: paragraph_p20_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_outline_idx ON public.paragraph_p20 USING btree (outline);


--
-- Name: paragraph_p20_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_pagenum_idx ON public.paragraph_p20 USING btree (pagenum);


--
-- Name: paragraph_p20_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_pdate_idx ON public.paragraph_p20 USING btree (pdate);


--
-- Name: paragraph_p20_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p20_source_url_source_page_pagenum_idx ON public.paragraph_p20 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p21_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_author_idx ON public.paragraph_p21 USING btree (author);


--
-- Name: paragraph_p21_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_dataset_idx ON public.paragraph_p21 USING btree (dataset);


--
-- Name: paragraph_p21_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_keywords_idx ON public.paragraph_p21 USING gin (keywords);


--
-- Name: paragraph_p21_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_outline_idx ON public.paragraph_p21 USING btree (outline);


--
-- Name: paragraph_p21_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_pagenum_idx ON public.paragraph_p21 USING btree (pagenum);


--
-- Name: paragraph_p21_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_pdate_idx ON public.paragraph_p21 USING btree (pdate);


--
-- Name: paragraph_p21_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p21_source_url_source_page_pagenum_idx ON public.paragraph_p21 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p22_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_author_idx ON public.paragraph_p22 USING btree (author);


--
-- Name: paragraph_p22_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_dataset_idx ON public.paragraph_p22 USING btree (dataset);


--
-- Name: paragraph_p22_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_keywords_idx ON public.paragraph_p22 USING gin (keywords);


--
-- Name: paragraph_p22_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_outline_idx ON public.paragraph_p22 USING btree (outline);


--
-- Name: paragraph_p22_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_pagenum_idx ON public.paragraph_p22 USING btree (pagenum);


--
-- Name: paragraph_p22_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_pdate_idx ON public.paragraph_p22 USING btree (pdate);


--
-- Name: paragraph_p22_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p22_source_url_source_page_pagenum_idx ON public.paragraph_p22 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p23_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_author_idx ON public.paragraph_p23 USING btree (author);


--
-- Name: paragraph_p23_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_dataset_idx ON public.paragraph_p23 USING btree (dataset);


--
-- Name: paragraph_p23_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_keywords_idx ON public.paragraph_p23 USING gin (keywords);


--
-- Name: paragraph_p23_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_outline_idx ON public.paragraph_p23 USING btree (outline);


--
-- Name: paragraph_p23_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_pagenum_idx ON public.paragraph_p23 USING btree (pagenum);


--
-- Name: paragraph_p23_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_pdate_idx ON public.paragraph_p23 USING btree (pdate);


--
-- Name: paragraph_p23_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p23_source_url_source_page_pagenum_idx ON public.paragraph_p23 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p24_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_author_idx ON public.paragraph_p24 USING btree (author);


--
-- Name: paragraph_p24_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_dataset_idx ON public.paragraph_p24 USING btree (dataset);


--
-- Name: paragraph_p24_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_keywords_idx ON public.paragraph_p24 USING gin (keywords);


--
-- Name: paragraph_p24_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_outline_idx ON public.paragraph_p24 USING btree (outline);


--
-- Name: paragraph_p24_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_pagenum_idx ON public.paragraph_p24 USING btree (pagenum);


--
-- Name: paragraph_p24_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_pdate_idx ON public.paragraph_p24 USING btree (pdate);


--
-- Name: paragraph_p24_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p24_source_url_source_page_pagenum_idx ON public.paragraph_p24 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p25_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_author_idx ON public.paragraph_p25 USING btree (author);


--
-- Name: paragraph_p25_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_dataset_idx ON public.paragraph_p25 USING btree (dataset);


--
-- Name: paragraph_p25_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_keywords_idx ON public.paragraph_p25 USING gin (keywords);


--
-- Name: paragraph_p25_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_outline_idx ON public.paragraph_p25 USING btree (outline);


--
-- Name: paragraph_p25_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_pagenum_idx ON public.paragraph_p25 USING btree (pagenum);


--
-- Name: paragraph_p25_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_pdate_idx ON public.paragraph_p25 USING btree (pdate);


--
-- Name: paragraph_p25_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p25_source_url_source_page_pagenum_idx ON public.paragraph_p25 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p26_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_author_idx ON public.paragraph_p26 USING btree (author);


--
-- Name: paragraph_p26_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_dataset_idx ON public.paragraph_p26 USING btree (dataset);


--
-- Name: paragraph_p26_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_keywords_idx ON public.paragraph_p26 USING gin (keywords);


--
-- Name: paragraph_p26_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_outline_idx ON public.paragraph_p26 USING btree (outline);


--
-- Name: paragraph_p26_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_pagenum_idx ON public.paragraph_p26 USING btree (pagenum);


--
-- Name: paragraph_p26_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_pdate_idx ON public.paragraph_p26 USING btree (pdate);


--
-- Name: paragraph_p26_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p26_source_url_source_page_pagenum_idx ON public.paragraph_p26 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p27_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_author_idx ON public.paragraph_p27 USING btree (author);


--
-- Name: paragraph_p27_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_dataset_idx ON public.paragraph_p27 USING btree (dataset);


--
-- Name: paragraph_p27_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_keywords_idx ON public.paragraph_p27 USING gin (keywords);


--
-- Name: paragraph_p27_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_outline_idx ON public.paragraph_p27 USING btree (outline);


--
-- Name: paragraph_p27_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_pagenum_idx ON public.paragraph_p27 USING btree (pagenum);


--
-- Name: paragraph_p27_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_pdate_idx ON public.paragraph_p27 USING btree (pdate);


--
-- Name: paragraph_p27_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p27_source_url_source_page_pagenum_idx ON public.paragraph_p27 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p28_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_author_idx ON public.paragraph_p28 USING btree (author);


--
-- Name: paragraph_p28_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_dataset_idx ON public.paragraph_p28 USING btree (dataset);


--
-- Name: paragraph_p28_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_keywords_idx ON public.paragraph_p28 USING gin (keywords);


--
-- Name: paragraph_p28_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_outline_idx ON public.paragraph_p28 USING btree (outline);


--
-- Name: paragraph_p28_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_pagenum_idx ON public.paragraph_p28 USING btree (pagenum);


--
-- Name: paragraph_p28_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_pdate_idx ON public.paragraph_p28 USING btree (pdate);


--
-- Name: paragraph_p28_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p28_source_url_source_page_pagenum_idx ON public.paragraph_p28 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p29_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_author_idx ON public.paragraph_p29 USING btree (author);


--
-- Name: paragraph_p29_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_dataset_idx ON public.paragraph_p29 USING btree (dataset);


--
-- Name: paragraph_p29_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_keywords_idx ON public.paragraph_p29 USING gin (keywords);


--
-- Name: paragraph_p29_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_outline_idx ON public.paragraph_p29 USING btree (outline);


--
-- Name: paragraph_p29_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_pagenum_idx ON public.paragraph_p29 USING btree (pagenum);


--
-- Name: paragraph_p29_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_pdate_idx ON public.paragraph_p29 USING btree (pdate);


--
-- Name: paragraph_p29_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p29_source_url_source_page_pagenum_idx ON public.paragraph_p29 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p2_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_author_idx ON public.paragraph_p2 USING btree (author);


--
-- Name: paragraph_p2_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_dataset_idx ON public.paragraph_p2 USING btree (dataset);


--
-- Name: paragraph_p2_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_keywords_idx ON public.paragraph_p2 USING gin (keywords);


--
-- Name: paragraph_p2_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_outline_idx ON public.paragraph_p2 USING btree (outline);


--
-- Name: paragraph_p2_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_pagenum_idx ON public.paragraph_p2 USING btree (pagenum);


--
-- Name: paragraph_p2_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_pdate_idx ON public.paragraph_p2 USING btree (pdate);


--
-- Name: paragraph_p2_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p2_source_url_source_page_pagenum_idx ON public.paragraph_p2 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p30_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_author_idx ON public.paragraph_p30 USING btree (author);


--
-- Name: paragraph_p30_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_dataset_idx ON public.paragraph_p30 USING btree (dataset);


--
-- Name: paragraph_p30_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_keywords_idx ON public.paragraph_p30 USING gin (keywords);


--
-- Name: paragraph_p30_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_outline_idx ON public.paragraph_p30 USING btree (outline);


--
-- Name: paragraph_p30_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_pagenum_idx ON public.paragraph_p30 USING btree (pagenum);


--
-- Name: paragraph_p30_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_pdate_idx ON public.paragraph_p30 USING btree (pdate);


--
-- Name: paragraph_p30_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p30_source_url_source_page_pagenum_idx ON public.paragraph_p30 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p31_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_author_idx ON public.paragraph_p31 USING btree (author);


--
-- Name: paragraph_p31_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_dataset_idx ON public.paragraph_p31 USING btree (dataset);


--
-- Name: paragraph_p31_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_keywords_idx ON public.paragraph_p31 USING gin (keywords);


--
-- Name: paragraph_p31_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_outline_idx ON public.paragraph_p31 USING btree (outline);


--
-- Name: paragraph_p31_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_pagenum_idx ON public.paragraph_p31 USING btree (pagenum);


--
-- Name: paragraph_p31_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_pdate_idx ON public.paragraph_p31 USING btree (pdate);


--
-- Name: paragraph_p31_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p31_source_url_source_page_pagenum_idx ON public.paragraph_p31 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p32_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_author_idx ON public.paragraph_p32 USING btree (author);


--
-- Name: paragraph_p32_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_dataset_idx ON public.paragraph_p32 USING btree (dataset);


--
-- Name: paragraph_p32_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_keywords_idx ON public.paragraph_p32 USING gin (keywords);


--
-- Name: paragraph_p32_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_outline_idx ON public.paragraph_p32 USING btree (outline);


--
-- Name: paragraph_p32_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_pagenum_idx ON public.paragraph_p32 USING btree (pagenum);


--
-- Name: paragraph_p32_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_pdate_idx ON public.paragraph_p32 USING btree (pdate);


--
-- Name: paragraph_p32_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p32_source_url_source_page_pagenum_idx ON public.paragraph_p32 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p33_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_author_idx ON public.paragraph_p33 USING btree (author);


--
-- Name: paragraph_p33_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_dataset_idx ON public.paragraph_p33 USING btree (dataset);


--
-- Name: paragraph_p33_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_keywords_idx ON public.paragraph_p33 USING gin (keywords);


--
-- Name: paragraph_p33_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_outline_idx ON public.paragraph_p33 USING btree (outline);


--
-- Name: paragraph_p33_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_pagenum_idx ON public.paragraph_p33 USING btree (pagenum);


--
-- Name: paragraph_p33_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_pdate_idx ON public.paragraph_p33 USING btree (pdate);


--
-- Name: paragraph_p33_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p33_source_url_source_page_pagenum_idx ON public.paragraph_p33 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p34_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_author_idx ON public.paragraph_p34 USING btree (author);


--
-- Name: paragraph_p34_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_dataset_idx ON public.paragraph_p34 USING btree (dataset);


--
-- Name: paragraph_p34_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_keywords_idx ON public.paragraph_p34 USING gin (keywords);


--
-- Name: paragraph_p34_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_outline_idx ON public.paragraph_p34 USING btree (outline);


--
-- Name: paragraph_p34_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_pagenum_idx ON public.paragraph_p34 USING btree (pagenum);


--
-- Name: paragraph_p34_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_pdate_idx ON public.paragraph_p34 USING btree (pdate);


--
-- Name: paragraph_p34_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p34_source_url_source_page_pagenum_idx ON public.paragraph_p34 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p35_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_author_idx ON public.paragraph_p35 USING btree (author);


--
-- Name: paragraph_p35_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_dataset_idx ON public.paragraph_p35 USING btree (dataset);


--
-- Name: paragraph_p35_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_keywords_idx ON public.paragraph_p35 USING gin (keywords);


--
-- Name: paragraph_p35_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_outline_idx ON public.paragraph_p35 USING btree (outline);


--
-- Name: paragraph_p35_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_pagenum_idx ON public.paragraph_p35 USING btree (pagenum);


--
-- Name: paragraph_p35_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_pdate_idx ON public.paragraph_p35 USING btree (pdate);


--
-- Name: paragraph_p35_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p35_source_url_source_page_pagenum_idx ON public.paragraph_p35 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p36_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_author_idx ON public.paragraph_p36 USING btree (author);


--
-- Name: paragraph_p36_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_dataset_idx ON public.paragraph_p36 USING btree (dataset);


--
-- Name: paragraph_p36_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_keywords_idx ON public.paragraph_p36 USING gin (keywords);


--
-- Name: paragraph_p36_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_outline_idx ON public.paragraph_p36 USING btree (outline);


--
-- Name: paragraph_p36_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_pagenum_idx ON public.paragraph_p36 USING btree (pagenum);


--
-- Name: paragraph_p36_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_pdate_idx ON public.paragraph_p36 USING btree (pdate);


--
-- Name: paragraph_p36_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p36_source_url_source_page_pagenum_idx ON public.paragraph_p36 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p37_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_author_idx ON public.paragraph_p37 USING btree (author);


--
-- Name: paragraph_p37_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_dataset_idx ON public.paragraph_p37 USING btree (dataset);


--
-- Name: paragraph_p37_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_keywords_idx ON public.paragraph_p37 USING gin (keywords);


--
-- Name: paragraph_p37_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_outline_idx ON public.paragraph_p37 USING btree (outline);


--
-- Name: paragraph_p37_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_pagenum_idx ON public.paragraph_p37 USING btree (pagenum);


--
-- Name: paragraph_p37_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_pdate_idx ON public.paragraph_p37 USING btree (pdate);


--
-- Name: paragraph_p37_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p37_source_url_source_page_pagenum_idx ON public.paragraph_p37 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p38_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_author_idx ON public.paragraph_p38 USING btree (author);


--
-- Name: paragraph_p38_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_dataset_idx ON public.paragraph_p38 USING btree (dataset);


--
-- Name: paragraph_p38_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_keywords_idx ON public.paragraph_p38 USING gin (keywords);


--
-- Name: paragraph_p38_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_outline_idx ON public.paragraph_p38 USING btree (outline);


--
-- Name: paragraph_p38_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_pagenum_idx ON public.paragraph_p38 USING btree (pagenum);


--
-- Name: paragraph_p38_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_pdate_idx ON public.paragraph_p38 USING btree (pdate);


--
-- Name: paragraph_p38_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p38_source_url_source_page_pagenum_idx ON public.paragraph_p38 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p39_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_author_idx ON public.paragraph_p39 USING btree (author);


--
-- Name: paragraph_p39_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_dataset_idx ON public.paragraph_p39 USING btree (dataset);


--
-- Name: paragraph_p39_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_keywords_idx ON public.paragraph_p39 USING gin (keywords);


--
-- Name: paragraph_p39_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_outline_idx ON public.paragraph_p39 USING btree (outline);


--
-- Name: paragraph_p39_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_pagenum_idx ON public.paragraph_p39 USING btree (pagenum);


--
-- Name: paragraph_p39_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_pdate_idx ON public.paragraph_p39 USING btree (pdate);


--
-- Name: paragraph_p39_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p39_source_url_source_page_pagenum_idx ON public.paragraph_p39 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p3_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_author_idx ON public.paragraph_p3 USING btree (author);


--
-- Name: paragraph_p3_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_dataset_idx ON public.paragraph_p3 USING btree (dataset);


--
-- Name: paragraph_p3_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_keywords_idx ON public.paragraph_p3 USING gin (keywords);


--
-- Name: paragraph_p3_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_outline_idx ON public.paragraph_p3 USING btree (outline);


--
-- Name: paragraph_p3_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_pagenum_idx ON public.paragraph_p3 USING btree (pagenum);


--
-- Name: paragraph_p3_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_pdate_idx ON public.paragraph_p3 USING btree (pdate);


--
-- Name: paragraph_p3_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p3_source_url_source_page_pagenum_idx ON public.paragraph_p3 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p40_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_author_idx ON public.paragraph_p40 USING btree (author);


--
-- Name: paragraph_p40_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_dataset_idx ON public.paragraph_p40 USING btree (dataset);


--
-- Name: paragraph_p40_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_keywords_idx ON public.paragraph_p40 USING gin (keywords);


--
-- Name: paragraph_p40_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_outline_idx ON public.paragraph_p40 USING btree (outline);


--
-- Name: paragraph_p40_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_pagenum_idx ON public.paragraph_p40 USING btree (pagenum);


--
-- Name: paragraph_p40_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_pdate_idx ON public.paragraph_p40 USING btree (pdate);


--
-- Name: paragraph_p40_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p40_source_url_source_page_pagenum_idx ON public.paragraph_p40 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p41_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_author_idx ON public.paragraph_p41 USING btree (author);


--
-- Name: paragraph_p41_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_dataset_idx ON public.paragraph_p41 USING btree (dataset);


--
-- Name: paragraph_p41_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_keywords_idx ON public.paragraph_p41 USING gin (keywords);


--
-- Name: paragraph_p41_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_outline_idx ON public.paragraph_p41 USING btree (outline);


--
-- Name: paragraph_p41_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_pagenum_idx ON public.paragraph_p41 USING btree (pagenum);


--
-- Name: paragraph_p41_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_pdate_idx ON public.paragraph_p41 USING btree (pdate);


--
-- Name: paragraph_p41_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p41_source_url_source_page_pagenum_idx ON public.paragraph_p41 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p42_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_author_idx ON public.paragraph_p42 USING btree (author);


--
-- Name: paragraph_p42_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_dataset_idx ON public.paragraph_p42 USING btree (dataset);


--
-- Name: paragraph_p42_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_keywords_idx ON public.paragraph_p42 USING gin (keywords);


--
-- Name: paragraph_p42_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_outline_idx ON public.paragraph_p42 USING btree (outline);


--
-- Name: paragraph_p42_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_pagenum_idx ON public.paragraph_p42 USING btree (pagenum);


--
-- Name: paragraph_p42_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_pdate_idx ON public.paragraph_p42 USING btree (pdate);


--
-- Name: paragraph_p42_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p42_source_url_source_page_pagenum_idx ON public.paragraph_p42 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p43_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_author_idx ON public.paragraph_p43 USING btree (author);


--
-- Name: paragraph_p43_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_dataset_idx ON public.paragraph_p43 USING btree (dataset);


--
-- Name: paragraph_p43_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_keywords_idx ON public.paragraph_p43 USING gin (keywords);


--
-- Name: paragraph_p43_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_outline_idx ON public.paragraph_p43 USING btree (outline);


--
-- Name: paragraph_p43_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_pagenum_idx ON public.paragraph_p43 USING btree (pagenum);


--
-- Name: paragraph_p43_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_pdate_idx ON public.paragraph_p43 USING btree (pdate);


--
-- Name: paragraph_p43_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p43_source_url_source_page_pagenum_idx ON public.paragraph_p43 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p44_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_author_idx ON public.paragraph_p44 USING btree (author);


--
-- Name: paragraph_p44_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_dataset_idx ON public.paragraph_p44 USING btree (dataset);


--
-- Name: paragraph_p44_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_keywords_idx ON public.paragraph_p44 USING gin (keywords);


--
-- Name: paragraph_p44_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_outline_idx ON public.paragraph_p44 USING btree (outline);


--
-- Name: paragraph_p44_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_pagenum_idx ON public.paragraph_p44 USING btree (pagenum);


--
-- Name: paragraph_p44_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_pdate_idx ON public.paragraph_p44 USING btree (pdate);


--
-- Name: paragraph_p44_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p44_source_url_source_page_pagenum_idx ON public.paragraph_p44 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p45_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_author_idx ON public.paragraph_p45 USING btree (author);


--
-- Name: paragraph_p45_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_dataset_idx ON public.paragraph_p45 USING btree (dataset);


--
-- Name: paragraph_p45_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_keywords_idx ON public.paragraph_p45 USING gin (keywords);


--
-- Name: paragraph_p45_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_outline_idx ON public.paragraph_p45 USING btree (outline);


--
-- Name: paragraph_p45_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_pagenum_idx ON public.paragraph_p45 USING btree (pagenum);


--
-- Name: paragraph_p45_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_pdate_idx ON public.paragraph_p45 USING btree (pdate);


--
-- Name: paragraph_p45_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p45_source_url_source_page_pagenum_idx ON public.paragraph_p45 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p46_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_author_idx ON public.paragraph_p46 USING btree (author);


--
-- Name: paragraph_p46_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_dataset_idx ON public.paragraph_p46 USING btree (dataset);


--
-- Name: paragraph_p46_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_keywords_idx ON public.paragraph_p46 USING gin (keywords);


--
-- Name: paragraph_p46_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_outline_idx ON public.paragraph_p46 USING btree (outline);


--
-- Name: paragraph_p46_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_pagenum_idx ON public.paragraph_p46 USING btree (pagenum);


--
-- Name: paragraph_p46_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_pdate_idx ON public.paragraph_p46 USING btree (pdate);


--
-- Name: paragraph_p46_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p46_source_url_source_page_pagenum_idx ON public.paragraph_p46 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p47_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_author_idx ON public.paragraph_p47 USING btree (author);


--
-- Name: paragraph_p47_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_dataset_idx ON public.paragraph_p47 USING btree (dataset);


--
-- Name: paragraph_p47_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_keywords_idx ON public.paragraph_p47 USING gin (keywords);


--
-- Name: paragraph_p47_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_outline_idx ON public.paragraph_p47 USING btree (outline);


--
-- Name: paragraph_p47_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_pagenum_idx ON public.paragraph_p47 USING btree (pagenum);


--
-- Name: paragraph_p47_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_pdate_idx ON public.paragraph_p47 USING btree (pdate);


--
-- Name: paragraph_p47_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p47_source_url_source_page_pagenum_idx ON public.paragraph_p47 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p48_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_author_idx ON public.paragraph_p48 USING btree (author);


--
-- Name: paragraph_p48_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_dataset_idx ON public.paragraph_p48 USING btree (dataset);


--
-- Name: paragraph_p48_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_keywords_idx ON public.paragraph_p48 USING gin (keywords);


--
-- Name: paragraph_p48_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_outline_idx ON public.paragraph_p48 USING btree (outline);


--
-- Name: paragraph_p48_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_pagenum_idx ON public.paragraph_p48 USING btree (pagenum);


--
-- Name: paragraph_p48_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_pdate_idx ON public.paragraph_p48 USING btree (pdate);


--
-- Name: paragraph_p48_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p48_source_url_source_page_pagenum_idx ON public.paragraph_p48 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p49_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_author_idx ON public.paragraph_p49 USING btree (author);


--
-- Name: paragraph_p49_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_dataset_idx ON public.paragraph_p49 USING btree (dataset);


--
-- Name: paragraph_p49_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_keywords_idx ON public.paragraph_p49 USING gin (keywords);


--
-- Name: paragraph_p49_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_outline_idx ON public.paragraph_p49 USING btree (outline);


--
-- Name: paragraph_p49_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_pagenum_idx ON public.paragraph_p49 USING btree (pagenum);


--
-- Name: paragraph_p49_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_pdate_idx ON public.paragraph_p49 USING btree (pdate);


--
-- Name: paragraph_p49_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p49_source_url_source_page_pagenum_idx ON public.paragraph_p49 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p4_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_author_idx ON public.paragraph_p4 USING btree (author);


--
-- Name: paragraph_p4_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_dataset_idx ON public.paragraph_p4 USING btree (dataset);


--
-- Name: paragraph_p4_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_keywords_idx ON public.paragraph_p4 USING gin (keywords);


--
-- Name: paragraph_p4_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_outline_idx ON public.paragraph_p4 USING btree (outline);


--
-- Name: paragraph_p4_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_pagenum_idx ON public.paragraph_p4 USING btree (pagenum);


--
-- Name: paragraph_p4_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_pdate_idx ON public.paragraph_p4 USING btree (pdate);


--
-- Name: paragraph_p4_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p4_source_url_source_page_pagenum_idx ON public.paragraph_p4 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p50_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_author_idx ON public.paragraph_p50 USING btree (author);


--
-- Name: paragraph_p50_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_dataset_idx ON public.paragraph_p50 USING btree (dataset);


--
-- Name: paragraph_p50_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_keywords_idx ON public.paragraph_p50 USING gin (keywords);


--
-- Name: paragraph_p50_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_outline_idx ON public.paragraph_p50 USING btree (outline);


--
-- Name: paragraph_p50_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_pagenum_idx ON public.paragraph_p50 USING btree (pagenum);


--
-- Name: paragraph_p50_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_pdate_idx ON public.paragraph_p50 USING btree (pdate);


--
-- Name: paragraph_p50_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p50_source_url_source_page_pagenum_idx ON public.paragraph_p50 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p51_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_author_idx ON public.paragraph_p51 USING btree (author);


--
-- Name: paragraph_p51_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_dataset_idx ON public.paragraph_p51 USING btree (dataset);


--
-- Name: paragraph_p51_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_keywords_idx ON public.paragraph_p51 USING gin (keywords);


--
-- Name: paragraph_p51_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_outline_idx ON public.paragraph_p51 USING btree (outline);


--
-- Name: paragraph_p51_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_pagenum_idx ON public.paragraph_p51 USING btree (pagenum);


--
-- Name: paragraph_p51_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_pdate_idx ON public.paragraph_p51 USING btree (pdate);


--
-- Name: paragraph_p51_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p51_source_url_source_page_pagenum_idx ON public.paragraph_p51 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p52_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_author_idx ON public.paragraph_p52 USING btree (author);


--
-- Name: paragraph_p52_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_dataset_idx ON public.paragraph_p52 USING btree (dataset);


--
-- Name: paragraph_p52_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_keywords_idx ON public.paragraph_p52 USING gin (keywords);


--
-- Name: paragraph_p52_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_outline_idx ON public.paragraph_p52 USING btree (outline);


--
-- Name: paragraph_p52_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_pagenum_idx ON public.paragraph_p52 USING btree (pagenum);


--
-- Name: paragraph_p52_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_pdate_idx ON public.paragraph_p52 USING btree (pdate);


--
-- Name: paragraph_p52_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p52_source_url_source_page_pagenum_idx ON public.paragraph_p52 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p53_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_author_idx ON public.paragraph_p53 USING btree (author);


--
-- Name: paragraph_p53_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_dataset_idx ON public.paragraph_p53 USING btree (dataset);


--
-- Name: paragraph_p53_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_keywords_idx ON public.paragraph_p53 USING gin (keywords);


--
-- Name: paragraph_p53_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_outline_idx ON public.paragraph_p53 USING btree (outline);


--
-- Name: paragraph_p53_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_pagenum_idx ON public.paragraph_p53 USING btree (pagenum);


--
-- Name: paragraph_p53_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_pdate_idx ON public.paragraph_p53 USING btree (pdate);


--
-- Name: paragraph_p53_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p53_source_url_source_page_pagenum_idx ON public.paragraph_p53 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p54_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_author_idx ON public.paragraph_p54 USING btree (author);


--
-- Name: paragraph_p54_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_dataset_idx ON public.paragraph_p54 USING btree (dataset);


--
-- Name: paragraph_p54_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_keywords_idx ON public.paragraph_p54 USING gin (keywords);


--
-- Name: paragraph_p54_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_outline_idx ON public.paragraph_p54 USING btree (outline);


--
-- Name: paragraph_p54_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_pagenum_idx ON public.paragraph_p54 USING btree (pagenum);


--
-- Name: paragraph_p54_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_pdate_idx ON public.paragraph_p54 USING btree (pdate);


--
-- Name: paragraph_p54_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p54_source_url_source_page_pagenum_idx ON public.paragraph_p54 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p55_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_author_idx ON public.paragraph_p55 USING btree (author);


--
-- Name: paragraph_p55_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_dataset_idx ON public.paragraph_p55 USING btree (dataset);


--
-- Name: paragraph_p55_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_keywords_idx ON public.paragraph_p55 USING gin (keywords);


--
-- Name: paragraph_p55_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_outline_idx ON public.paragraph_p55 USING btree (outline);


--
-- Name: paragraph_p55_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_pagenum_idx ON public.paragraph_p55 USING btree (pagenum);


--
-- Name: paragraph_p55_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_pdate_idx ON public.paragraph_p55 USING btree (pdate);


--
-- Name: paragraph_p55_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p55_source_url_source_page_pagenum_idx ON public.paragraph_p55 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p56_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_author_idx ON public.paragraph_p56 USING btree (author);


--
-- Name: paragraph_p56_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_dataset_idx ON public.paragraph_p56 USING btree (dataset);


--
-- Name: paragraph_p56_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_keywords_idx ON public.paragraph_p56 USING gin (keywords);


--
-- Name: paragraph_p56_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_outline_idx ON public.paragraph_p56 USING btree (outline);


--
-- Name: paragraph_p56_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_pagenum_idx ON public.paragraph_p56 USING btree (pagenum);


--
-- Name: paragraph_p56_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_pdate_idx ON public.paragraph_p56 USING btree (pdate);


--
-- Name: paragraph_p56_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p56_source_url_source_page_pagenum_idx ON public.paragraph_p56 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p57_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_author_idx ON public.paragraph_p57 USING btree (author);


--
-- Name: paragraph_p57_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_dataset_idx ON public.paragraph_p57 USING btree (dataset);


--
-- Name: paragraph_p57_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_keywords_idx ON public.paragraph_p57 USING gin (keywords);


--
-- Name: paragraph_p57_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_outline_idx ON public.paragraph_p57 USING btree (outline);


--
-- Name: paragraph_p57_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_pagenum_idx ON public.paragraph_p57 USING btree (pagenum);


--
-- Name: paragraph_p57_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_pdate_idx ON public.paragraph_p57 USING btree (pdate);


--
-- Name: paragraph_p57_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p57_source_url_source_page_pagenum_idx ON public.paragraph_p57 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p58_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_author_idx ON public.paragraph_p58 USING btree (author);


--
-- Name: paragraph_p58_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_dataset_idx ON public.paragraph_p58 USING btree (dataset);


--
-- Name: paragraph_p58_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_keywords_idx ON public.paragraph_p58 USING gin (keywords);


--
-- Name: paragraph_p58_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_outline_idx ON public.paragraph_p58 USING btree (outline);


--
-- Name: paragraph_p58_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_pagenum_idx ON public.paragraph_p58 USING btree (pagenum);


--
-- Name: paragraph_p58_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_pdate_idx ON public.paragraph_p58 USING btree (pdate);


--
-- Name: paragraph_p58_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p58_source_url_source_page_pagenum_idx ON public.paragraph_p58 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p59_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_author_idx ON public.paragraph_p59 USING btree (author);


--
-- Name: paragraph_p59_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_dataset_idx ON public.paragraph_p59 USING btree (dataset);


--
-- Name: paragraph_p59_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_keywords_idx ON public.paragraph_p59 USING gin (keywords);


--
-- Name: paragraph_p59_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_outline_idx ON public.paragraph_p59 USING btree (outline);


--
-- Name: paragraph_p59_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_pagenum_idx ON public.paragraph_p59 USING btree (pagenum);


--
-- Name: paragraph_p59_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_pdate_idx ON public.paragraph_p59 USING btree (pdate);


--
-- Name: paragraph_p59_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p59_source_url_source_page_pagenum_idx ON public.paragraph_p59 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p5_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_author_idx ON public.paragraph_p5 USING btree (author);


--
-- Name: paragraph_p5_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_dataset_idx ON public.paragraph_p5 USING btree (dataset);


--
-- Name: paragraph_p5_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_keywords_idx ON public.paragraph_p5 USING gin (keywords);


--
-- Name: paragraph_p5_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_outline_idx ON public.paragraph_p5 USING btree (outline);


--
-- Name: paragraph_p5_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_pagenum_idx ON public.paragraph_p5 USING btree (pagenum);


--
-- Name: paragraph_p5_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_pdate_idx ON public.paragraph_p5 USING btree (pdate);


--
-- Name: paragraph_p5_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p5_source_url_source_page_pagenum_idx ON public.paragraph_p5 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p60_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_author_idx ON public.paragraph_p60 USING btree (author);


--
-- Name: paragraph_p60_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_dataset_idx ON public.paragraph_p60 USING btree (dataset);


--
-- Name: paragraph_p60_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_keywords_idx ON public.paragraph_p60 USING gin (keywords);


--
-- Name: paragraph_p60_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_outline_idx ON public.paragraph_p60 USING btree (outline);


--
-- Name: paragraph_p60_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_pagenum_idx ON public.paragraph_p60 USING btree (pagenum);


--
-- Name: paragraph_p60_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_pdate_idx ON public.paragraph_p60 USING btree (pdate);


--
-- Name: paragraph_p60_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p60_source_url_source_page_pagenum_idx ON public.paragraph_p60 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p61_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_author_idx ON public.paragraph_p61 USING btree (author);


--
-- Name: paragraph_p61_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_dataset_idx ON public.paragraph_p61 USING btree (dataset);


--
-- Name: paragraph_p61_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_keywords_idx ON public.paragraph_p61 USING gin (keywords);


--
-- Name: paragraph_p61_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_outline_idx ON public.paragraph_p61 USING btree (outline);


--
-- Name: paragraph_p61_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_pagenum_idx ON public.paragraph_p61 USING btree (pagenum);


--
-- Name: paragraph_p61_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_pdate_idx ON public.paragraph_p61 USING btree (pdate);


--
-- Name: paragraph_p61_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p61_source_url_source_page_pagenum_idx ON public.paragraph_p61 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p62_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_author_idx ON public.paragraph_p62 USING btree (author);


--
-- Name: paragraph_p62_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_dataset_idx ON public.paragraph_p62 USING btree (dataset);


--
-- Name: paragraph_p62_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_keywords_idx ON public.paragraph_p62 USING gin (keywords);


--
-- Name: paragraph_p62_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_outline_idx ON public.paragraph_p62 USING btree (outline);


--
-- Name: paragraph_p62_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_pagenum_idx ON public.paragraph_p62 USING btree (pagenum);


--
-- Name: paragraph_p62_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_pdate_idx ON public.paragraph_p62 USING btree (pdate);


--
-- Name: paragraph_p62_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p62_source_url_source_page_pagenum_idx ON public.paragraph_p62 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p63_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_author_idx ON public.paragraph_p63 USING btree (author);


--
-- Name: paragraph_p63_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_dataset_idx ON public.paragraph_p63 USING btree (dataset);


--
-- Name: paragraph_p63_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_keywords_idx ON public.paragraph_p63 USING gin (keywords);


--
-- Name: paragraph_p63_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_outline_idx ON public.paragraph_p63 USING btree (outline);


--
-- Name: paragraph_p63_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_pagenum_idx ON public.paragraph_p63 USING btree (pagenum);


--
-- Name: paragraph_p63_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_pdate_idx ON public.paragraph_p63 USING btree (pdate);


--
-- Name: paragraph_p63_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p63_source_url_source_page_pagenum_idx ON public.paragraph_p63 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p6_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_author_idx ON public.paragraph_p6 USING btree (author);


--
-- Name: paragraph_p6_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_dataset_idx ON public.paragraph_p6 USING btree (dataset);


--
-- Name: paragraph_p6_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_keywords_idx ON public.paragraph_p6 USING gin (keywords);


--
-- Name: paragraph_p6_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_outline_idx ON public.paragraph_p6 USING btree (outline);


--
-- Name: paragraph_p6_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_pagenum_idx ON public.paragraph_p6 USING btree (pagenum);


--
-- Name: paragraph_p6_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_pdate_idx ON public.paragraph_p6 USING btree (pdate);


--
-- Name: paragraph_p6_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p6_source_url_source_page_pagenum_idx ON public.paragraph_p6 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p7_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_author_idx ON public.paragraph_p7 USING btree (author);


--
-- Name: paragraph_p7_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_dataset_idx ON public.paragraph_p7 USING btree (dataset);


--
-- Name: paragraph_p7_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_keywords_idx ON public.paragraph_p7 USING gin (keywords);


--
-- Name: paragraph_p7_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_outline_idx ON public.paragraph_p7 USING btree (outline);


--
-- Name: paragraph_p7_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_pagenum_idx ON public.paragraph_p7 USING btree (pagenum);


--
-- Name: paragraph_p7_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_pdate_idx ON public.paragraph_p7 USING btree (pdate);


--
-- Name: paragraph_p7_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p7_source_url_source_page_pagenum_idx ON public.paragraph_p7 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p8_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_author_idx ON public.paragraph_p8 USING btree (author);


--
-- Name: paragraph_p8_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_dataset_idx ON public.paragraph_p8 USING btree (dataset);


--
-- Name: paragraph_p8_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_keywords_idx ON public.paragraph_p8 USING gin (keywords);


--
-- Name: paragraph_p8_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_outline_idx ON public.paragraph_p8 USING btree (outline);


--
-- Name: paragraph_p8_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_pagenum_idx ON public.paragraph_p8 USING btree (pagenum);


--
-- Name: paragraph_p8_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_pdate_idx ON public.paragraph_p8 USING btree (pdate);


--
-- Name: paragraph_p8_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p8_source_url_source_page_pagenum_idx ON public.paragraph_p8 USING btree (source_url, source_page, pagenum);


--
-- Name: paragraph_p9_author_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_author_idx ON public.paragraph_p9 USING btree (author);


--
-- Name: paragraph_p9_dataset_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_dataset_idx ON public.paragraph_p9 USING btree (dataset);


--
-- Name: paragraph_p9_keywords_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_keywords_idx ON public.paragraph_p9 USING gin (keywords);


--
-- Name: paragraph_p9_outline_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_outline_idx ON public.paragraph_p9 USING btree (outline);


--
-- Name: paragraph_p9_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_pagenum_idx ON public.paragraph_p9 USING btree (pagenum);


--
-- Name: paragraph_p9_pdate_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_pdate_idx ON public.paragraph_p9 USING btree (pdate);


--
-- Name: paragraph_p9_source_url_source_page_pagenum_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX paragraph_p9_source_url_source_page_pagenum_idx ON public.paragraph_p9 USING btree (source_url, source_page, pagenum);


--
-- Name: text_embeddings_p0_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p0_embedding_idx ON public.text_embeddings_p0 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p10_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p10_embedding_idx ON public.text_embeddings_p10 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p11_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p11_embedding_idx ON public.text_embeddings_p11 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p12_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p12_embedding_idx ON public.text_embeddings_p12 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p13_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p13_embedding_idx ON public.text_embeddings_p13 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p14_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p14_embedding_idx ON public.text_embeddings_p14 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p15_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p15_embedding_idx ON public.text_embeddings_p15 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p16_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p16_embedding_idx ON public.text_embeddings_p16 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p17_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p17_embedding_idx ON public.text_embeddings_p17 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p18_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p18_embedding_idx ON public.text_embeddings_p18 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p19_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p19_embedding_idx ON public.text_embeddings_p19 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p1_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p1_embedding_idx ON public.text_embeddings_p1 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p20_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p20_embedding_idx ON public.text_embeddings_p20 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p21_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p21_embedding_idx ON public.text_embeddings_p21 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p22_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p22_embedding_idx ON public.text_embeddings_p22 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p23_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p23_embedding_idx ON public.text_embeddings_p23 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p24_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p24_embedding_idx ON public.text_embeddings_p24 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p25_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p25_embedding_idx ON public.text_embeddings_p25 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p26_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p26_embedding_idx ON public.text_embeddings_p26 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p27_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p27_embedding_idx ON public.text_embeddings_p27 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p28_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p28_embedding_idx ON public.text_embeddings_p28 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p29_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p29_embedding_idx ON public.text_embeddings_p29 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p2_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p2_embedding_idx ON public.text_embeddings_p2 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p30_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p30_embedding_idx ON public.text_embeddings_p30 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p31_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p31_embedding_idx ON public.text_embeddings_p31 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p32_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p32_embedding_idx ON public.text_embeddings_p32 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p33_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p33_embedding_idx ON public.text_embeddings_p33 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p34_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p34_embedding_idx ON public.text_embeddings_p34 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p35_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p35_embedding_idx ON public.text_embeddings_p35 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p36_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p36_embedding_idx ON public.text_embeddings_p36 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p37_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p37_embedding_idx ON public.text_embeddings_p37 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p38_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p38_embedding_idx ON public.text_embeddings_p38 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p39_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p39_embedding_idx ON public.text_embeddings_p39 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p3_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p3_embedding_idx ON public.text_embeddings_p3 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p40_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p40_embedding_idx ON public.text_embeddings_p40 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p41_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p41_embedding_idx ON public.text_embeddings_p41 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p42_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p42_embedding_idx ON public.text_embeddings_p42 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p43_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p43_embedding_idx ON public.text_embeddings_p43 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p44_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p44_embedding_idx ON public.text_embeddings_p44 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p45_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p45_embedding_idx ON public.text_embeddings_p45 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p46_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p46_embedding_idx ON public.text_embeddings_p46 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p47_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p47_embedding_idx ON public.text_embeddings_p47 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p48_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p48_embedding_idx ON public.text_embeddings_p48 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p49_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p49_embedding_idx ON public.text_embeddings_p49 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p4_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p4_embedding_idx ON public.text_embeddings_p4 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p50_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p50_embedding_idx ON public.text_embeddings_p50 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p51_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p51_embedding_idx ON public.text_embeddings_p51 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p52_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p52_embedding_idx ON public.text_embeddings_p52 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p53_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p53_embedding_idx ON public.text_embeddings_p53 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p54_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p54_embedding_idx ON public.text_embeddings_p54 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p55_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p55_embedding_idx ON public.text_embeddings_p55 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p56_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p56_embedding_idx ON public.text_embeddings_p56 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p57_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p57_embedding_idx ON public.text_embeddings_p57 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p58_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p58_embedding_idx ON public.text_embeddings_p58 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p59_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p59_embedding_idx ON public.text_embeddings_p59 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p5_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p5_embedding_idx ON public.text_embeddings_p5 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p60_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p60_embedding_idx ON public.text_embeddings_p60 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p61_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p61_embedding_idx ON public.text_embeddings_p61 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p62_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p62_embedding_idx ON public.text_embeddings_p62 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p63_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p63_embedding_idx ON public.text_embeddings_p63 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p6_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p6_embedding_idx ON public.text_embeddings_p6 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p7_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p7_embedding_idx ON public.text_embeddings_p7 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p8_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p8_embedding_idx ON public.text_embeddings_p8 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: text_embeddings_p9_embedding_idx; Type: INDEX; Schema: public; Owner: myuser
--

CREATE INDEX text_embeddings_p9_embedding_idx ON public.text_embeddings_p9 USING vchordrq (embedding public.halfvec_cosine_ops);


--
-- Name: paragraph_p0_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p0_author_idx;


--
-- Name: paragraph_p0_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p0_dataset_idx;


--
-- Name: paragraph_p0_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p0_keywords_idx;


--
-- Name: paragraph_p0_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p0_outline_idx;


--
-- Name: paragraph_p0_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p0_pagenum_idx;


--
-- Name: paragraph_p0_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p0_pdate_idx;


--
-- Name: paragraph_p0_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p0_pkey;


--
-- Name: paragraph_p0_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p0_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p10_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p10_author_idx;


--
-- Name: paragraph_p10_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p10_dataset_idx;


--
-- Name: paragraph_p10_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p10_keywords_idx;


--
-- Name: paragraph_p10_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p10_outline_idx;


--
-- Name: paragraph_p10_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p10_pagenum_idx;


--
-- Name: paragraph_p10_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p10_pdate_idx;


--
-- Name: paragraph_p10_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p10_pkey;


--
-- Name: paragraph_p10_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p10_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p11_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p11_author_idx;


--
-- Name: paragraph_p11_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p11_dataset_idx;


--
-- Name: paragraph_p11_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p11_keywords_idx;


--
-- Name: paragraph_p11_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p11_outline_idx;


--
-- Name: paragraph_p11_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p11_pagenum_idx;


--
-- Name: paragraph_p11_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p11_pdate_idx;


--
-- Name: paragraph_p11_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p11_pkey;


--
-- Name: paragraph_p11_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p11_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p12_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p12_author_idx;


--
-- Name: paragraph_p12_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p12_dataset_idx;


--
-- Name: paragraph_p12_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p12_keywords_idx;


--
-- Name: paragraph_p12_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p12_outline_idx;


--
-- Name: paragraph_p12_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p12_pagenum_idx;


--
-- Name: paragraph_p12_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p12_pdate_idx;


--
-- Name: paragraph_p12_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p12_pkey;


--
-- Name: paragraph_p12_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p12_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p13_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p13_author_idx;


--
-- Name: paragraph_p13_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p13_dataset_idx;


--
-- Name: paragraph_p13_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p13_keywords_idx;


--
-- Name: paragraph_p13_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p13_outline_idx;


--
-- Name: paragraph_p13_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p13_pagenum_idx;


--
-- Name: paragraph_p13_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p13_pdate_idx;


--
-- Name: paragraph_p13_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p13_pkey;


--
-- Name: paragraph_p13_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p13_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p14_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p14_author_idx;


--
-- Name: paragraph_p14_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p14_dataset_idx;


--
-- Name: paragraph_p14_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p14_keywords_idx;


--
-- Name: paragraph_p14_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p14_outline_idx;


--
-- Name: paragraph_p14_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p14_pagenum_idx;


--
-- Name: paragraph_p14_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p14_pdate_idx;


--
-- Name: paragraph_p14_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p14_pkey;


--
-- Name: paragraph_p14_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p14_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p15_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p15_author_idx;


--
-- Name: paragraph_p15_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p15_dataset_idx;


--
-- Name: paragraph_p15_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p15_keywords_idx;


--
-- Name: paragraph_p15_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p15_outline_idx;


--
-- Name: paragraph_p15_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p15_pagenum_idx;


--
-- Name: paragraph_p15_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p15_pdate_idx;


--
-- Name: paragraph_p15_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p15_pkey;


--
-- Name: paragraph_p15_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p15_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p16_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p16_author_idx;


--
-- Name: paragraph_p16_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p16_dataset_idx;


--
-- Name: paragraph_p16_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p16_keywords_idx;


--
-- Name: paragraph_p16_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p16_outline_idx;


--
-- Name: paragraph_p16_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p16_pagenum_idx;


--
-- Name: paragraph_p16_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p16_pdate_idx;


--
-- Name: paragraph_p16_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p16_pkey;


--
-- Name: paragraph_p16_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p16_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p17_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p17_author_idx;


--
-- Name: paragraph_p17_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p17_dataset_idx;


--
-- Name: paragraph_p17_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p17_keywords_idx;


--
-- Name: paragraph_p17_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p17_outline_idx;


--
-- Name: paragraph_p17_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p17_pagenum_idx;


--
-- Name: paragraph_p17_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p17_pdate_idx;


--
-- Name: paragraph_p17_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p17_pkey;


--
-- Name: paragraph_p17_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p17_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p18_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p18_author_idx;


--
-- Name: paragraph_p18_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p18_dataset_idx;


--
-- Name: paragraph_p18_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p18_keywords_idx;


--
-- Name: paragraph_p18_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p18_outline_idx;


--
-- Name: paragraph_p18_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p18_pagenum_idx;


--
-- Name: paragraph_p18_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p18_pdate_idx;


--
-- Name: paragraph_p18_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p18_pkey;


--
-- Name: paragraph_p18_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p18_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p19_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p19_author_idx;


--
-- Name: paragraph_p19_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p19_dataset_idx;


--
-- Name: paragraph_p19_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p19_keywords_idx;


--
-- Name: paragraph_p19_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p19_outline_idx;


--
-- Name: paragraph_p19_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p19_pagenum_idx;


--
-- Name: paragraph_p19_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p19_pdate_idx;


--
-- Name: paragraph_p19_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p19_pkey;


--
-- Name: paragraph_p19_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p19_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p1_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p1_author_idx;


--
-- Name: paragraph_p1_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p1_dataset_idx;


--
-- Name: paragraph_p1_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p1_keywords_idx;


--
-- Name: paragraph_p1_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p1_outline_idx;


--
-- Name: paragraph_p1_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p1_pagenum_idx;


--
-- Name: paragraph_p1_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p1_pdate_idx;


--
-- Name: paragraph_p1_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p1_pkey;


--
-- Name: paragraph_p1_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p1_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p20_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p20_author_idx;


--
-- Name: paragraph_p20_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p20_dataset_idx;


--
-- Name: paragraph_p20_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p20_keywords_idx;


--
-- Name: paragraph_p20_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p20_outline_idx;


--
-- Name: paragraph_p20_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p20_pagenum_idx;


--
-- Name: paragraph_p20_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p20_pdate_idx;


--
-- Name: paragraph_p20_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p20_pkey;


--
-- Name: paragraph_p20_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p20_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p21_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p21_author_idx;


--
-- Name: paragraph_p21_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p21_dataset_idx;


--
-- Name: paragraph_p21_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p21_keywords_idx;


--
-- Name: paragraph_p21_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p21_outline_idx;


--
-- Name: paragraph_p21_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p21_pagenum_idx;


--
-- Name: paragraph_p21_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p21_pdate_idx;


--
-- Name: paragraph_p21_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p21_pkey;


--
-- Name: paragraph_p21_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p21_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p22_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p22_author_idx;


--
-- Name: paragraph_p22_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p22_dataset_idx;


--
-- Name: paragraph_p22_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p22_keywords_idx;


--
-- Name: paragraph_p22_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p22_outline_idx;


--
-- Name: paragraph_p22_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p22_pagenum_idx;


--
-- Name: paragraph_p22_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p22_pdate_idx;


--
-- Name: paragraph_p22_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p22_pkey;


--
-- Name: paragraph_p22_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p22_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p23_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p23_author_idx;


--
-- Name: paragraph_p23_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p23_dataset_idx;


--
-- Name: paragraph_p23_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p23_keywords_idx;


--
-- Name: paragraph_p23_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p23_outline_idx;


--
-- Name: paragraph_p23_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p23_pagenum_idx;


--
-- Name: paragraph_p23_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p23_pdate_idx;


--
-- Name: paragraph_p23_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p23_pkey;


--
-- Name: paragraph_p23_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p23_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p24_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p24_author_idx;


--
-- Name: paragraph_p24_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p24_dataset_idx;


--
-- Name: paragraph_p24_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p24_keywords_idx;


--
-- Name: paragraph_p24_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p24_outline_idx;


--
-- Name: paragraph_p24_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p24_pagenum_idx;


--
-- Name: paragraph_p24_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p24_pdate_idx;


--
-- Name: paragraph_p24_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p24_pkey;


--
-- Name: paragraph_p24_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p24_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p25_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p25_author_idx;


--
-- Name: paragraph_p25_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p25_dataset_idx;


--
-- Name: paragraph_p25_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p25_keywords_idx;


--
-- Name: paragraph_p25_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p25_outline_idx;


--
-- Name: paragraph_p25_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p25_pagenum_idx;


--
-- Name: paragraph_p25_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p25_pdate_idx;


--
-- Name: paragraph_p25_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p25_pkey;


--
-- Name: paragraph_p25_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p25_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p26_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p26_author_idx;


--
-- Name: paragraph_p26_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p26_dataset_idx;


--
-- Name: paragraph_p26_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p26_keywords_idx;


--
-- Name: paragraph_p26_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p26_outline_idx;


--
-- Name: paragraph_p26_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p26_pagenum_idx;


--
-- Name: paragraph_p26_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p26_pdate_idx;


--
-- Name: paragraph_p26_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p26_pkey;


--
-- Name: paragraph_p26_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p26_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p27_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p27_author_idx;


--
-- Name: paragraph_p27_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p27_dataset_idx;


--
-- Name: paragraph_p27_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p27_keywords_idx;


--
-- Name: paragraph_p27_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p27_outline_idx;


--
-- Name: paragraph_p27_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p27_pagenum_idx;


--
-- Name: paragraph_p27_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p27_pdate_idx;


--
-- Name: paragraph_p27_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p27_pkey;


--
-- Name: paragraph_p27_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p27_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p28_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p28_author_idx;


--
-- Name: paragraph_p28_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p28_dataset_idx;


--
-- Name: paragraph_p28_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p28_keywords_idx;


--
-- Name: paragraph_p28_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p28_outline_idx;


--
-- Name: paragraph_p28_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p28_pagenum_idx;


--
-- Name: paragraph_p28_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p28_pdate_idx;


--
-- Name: paragraph_p28_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p28_pkey;


--
-- Name: paragraph_p28_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p28_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p29_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p29_author_idx;


--
-- Name: paragraph_p29_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p29_dataset_idx;


--
-- Name: paragraph_p29_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p29_keywords_idx;


--
-- Name: paragraph_p29_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p29_outline_idx;


--
-- Name: paragraph_p29_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p29_pagenum_idx;


--
-- Name: paragraph_p29_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p29_pdate_idx;


--
-- Name: paragraph_p29_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p29_pkey;


--
-- Name: paragraph_p29_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p29_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p2_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p2_author_idx;


--
-- Name: paragraph_p2_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p2_dataset_idx;


--
-- Name: paragraph_p2_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p2_keywords_idx;


--
-- Name: paragraph_p2_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p2_outline_idx;


--
-- Name: paragraph_p2_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p2_pagenum_idx;


--
-- Name: paragraph_p2_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p2_pdate_idx;


--
-- Name: paragraph_p2_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p2_pkey;


--
-- Name: paragraph_p2_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p2_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p30_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p30_author_idx;


--
-- Name: paragraph_p30_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p30_dataset_idx;


--
-- Name: paragraph_p30_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p30_keywords_idx;


--
-- Name: paragraph_p30_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p30_outline_idx;


--
-- Name: paragraph_p30_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p30_pagenum_idx;


--
-- Name: paragraph_p30_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p30_pdate_idx;


--
-- Name: paragraph_p30_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p30_pkey;


--
-- Name: paragraph_p30_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p30_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p31_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p31_author_idx;


--
-- Name: paragraph_p31_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p31_dataset_idx;


--
-- Name: paragraph_p31_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p31_keywords_idx;


--
-- Name: paragraph_p31_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p31_outline_idx;


--
-- Name: paragraph_p31_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p31_pagenum_idx;


--
-- Name: paragraph_p31_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p31_pdate_idx;


--
-- Name: paragraph_p31_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p31_pkey;


--
-- Name: paragraph_p31_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p31_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p32_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p32_author_idx;


--
-- Name: paragraph_p32_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p32_dataset_idx;


--
-- Name: paragraph_p32_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p32_keywords_idx;


--
-- Name: paragraph_p32_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p32_outline_idx;


--
-- Name: paragraph_p32_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p32_pagenum_idx;


--
-- Name: paragraph_p32_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p32_pdate_idx;


--
-- Name: paragraph_p32_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p32_pkey;


--
-- Name: paragraph_p32_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p32_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p33_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p33_author_idx;


--
-- Name: paragraph_p33_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p33_dataset_idx;


--
-- Name: paragraph_p33_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p33_keywords_idx;


--
-- Name: paragraph_p33_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p33_outline_idx;


--
-- Name: paragraph_p33_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p33_pagenum_idx;


--
-- Name: paragraph_p33_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p33_pdate_idx;


--
-- Name: paragraph_p33_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p33_pkey;


--
-- Name: paragraph_p33_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p33_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p34_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p34_author_idx;


--
-- Name: paragraph_p34_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p34_dataset_idx;


--
-- Name: paragraph_p34_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p34_keywords_idx;


--
-- Name: paragraph_p34_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p34_outline_idx;


--
-- Name: paragraph_p34_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p34_pagenum_idx;


--
-- Name: paragraph_p34_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p34_pdate_idx;


--
-- Name: paragraph_p34_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p34_pkey;


--
-- Name: paragraph_p34_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p34_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p35_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p35_author_idx;


--
-- Name: paragraph_p35_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p35_dataset_idx;


--
-- Name: paragraph_p35_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p35_keywords_idx;


--
-- Name: paragraph_p35_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p35_outline_idx;


--
-- Name: paragraph_p35_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p35_pagenum_idx;


--
-- Name: paragraph_p35_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p35_pdate_idx;


--
-- Name: paragraph_p35_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p35_pkey;


--
-- Name: paragraph_p35_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p35_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p36_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p36_author_idx;


--
-- Name: paragraph_p36_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p36_dataset_idx;


--
-- Name: paragraph_p36_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p36_keywords_idx;


--
-- Name: paragraph_p36_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p36_outline_idx;


--
-- Name: paragraph_p36_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p36_pagenum_idx;


--
-- Name: paragraph_p36_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p36_pdate_idx;


--
-- Name: paragraph_p36_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p36_pkey;


--
-- Name: paragraph_p36_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p36_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p37_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p37_author_idx;


--
-- Name: paragraph_p37_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p37_dataset_idx;


--
-- Name: paragraph_p37_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p37_keywords_idx;


--
-- Name: paragraph_p37_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p37_outline_idx;


--
-- Name: paragraph_p37_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p37_pagenum_idx;


--
-- Name: paragraph_p37_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p37_pdate_idx;


--
-- Name: paragraph_p37_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p37_pkey;


--
-- Name: paragraph_p37_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p37_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p38_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p38_author_idx;


--
-- Name: paragraph_p38_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p38_dataset_idx;


--
-- Name: paragraph_p38_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p38_keywords_idx;


--
-- Name: paragraph_p38_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p38_outline_idx;


--
-- Name: paragraph_p38_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p38_pagenum_idx;


--
-- Name: paragraph_p38_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p38_pdate_idx;


--
-- Name: paragraph_p38_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p38_pkey;


--
-- Name: paragraph_p38_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p38_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p39_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p39_author_idx;


--
-- Name: paragraph_p39_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p39_dataset_idx;


--
-- Name: paragraph_p39_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p39_keywords_idx;


--
-- Name: paragraph_p39_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p39_outline_idx;


--
-- Name: paragraph_p39_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p39_pagenum_idx;


--
-- Name: paragraph_p39_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p39_pdate_idx;


--
-- Name: paragraph_p39_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p39_pkey;


--
-- Name: paragraph_p39_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p39_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p3_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p3_author_idx;


--
-- Name: paragraph_p3_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p3_dataset_idx;


--
-- Name: paragraph_p3_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p3_keywords_idx;


--
-- Name: paragraph_p3_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p3_outline_idx;


--
-- Name: paragraph_p3_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p3_pagenum_idx;


--
-- Name: paragraph_p3_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p3_pdate_idx;


--
-- Name: paragraph_p3_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p3_pkey;


--
-- Name: paragraph_p3_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p3_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p40_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p40_author_idx;


--
-- Name: paragraph_p40_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p40_dataset_idx;


--
-- Name: paragraph_p40_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p40_keywords_idx;


--
-- Name: paragraph_p40_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p40_outline_idx;


--
-- Name: paragraph_p40_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p40_pagenum_idx;


--
-- Name: paragraph_p40_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p40_pdate_idx;


--
-- Name: paragraph_p40_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p40_pkey;


--
-- Name: paragraph_p40_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p40_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p41_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p41_author_idx;


--
-- Name: paragraph_p41_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p41_dataset_idx;


--
-- Name: paragraph_p41_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p41_keywords_idx;


--
-- Name: paragraph_p41_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p41_outline_idx;


--
-- Name: paragraph_p41_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p41_pagenum_idx;


--
-- Name: paragraph_p41_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p41_pdate_idx;


--
-- Name: paragraph_p41_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p41_pkey;


--
-- Name: paragraph_p41_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p41_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p42_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p42_author_idx;


--
-- Name: paragraph_p42_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p42_dataset_idx;


--
-- Name: paragraph_p42_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p42_keywords_idx;


--
-- Name: paragraph_p42_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p42_outline_idx;


--
-- Name: paragraph_p42_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p42_pagenum_idx;


--
-- Name: paragraph_p42_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p42_pdate_idx;


--
-- Name: paragraph_p42_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p42_pkey;


--
-- Name: paragraph_p42_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p42_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p43_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p43_author_idx;


--
-- Name: paragraph_p43_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p43_dataset_idx;


--
-- Name: paragraph_p43_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p43_keywords_idx;


--
-- Name: paragraph_p43_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p43_outline_idx;


--
-- Name: paragraph_p43_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p43_pagenum_idx;


--
-- Name: paragraph_p43_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p43_pdate_idx;


--
-- Name: paragraph_p43_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p43_pkey;


--
-- Name: paragraph_p43_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p43_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p44_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p44_author_idx;


--
-- Name: paragraph_p44_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p44_dataset_idx;


--
-- Name: paragraph_p44_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p44_keywords_idx;


--
-- Name: paragraph_p44_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p44_outline_idx;


--
-- Name: paragraph_p44_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p44_pagenum_idx;


--
-- Name: paragraph_p44_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p44_pdate_idx;


--
-- Name: paragraph_p44_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p44_pkey;


--
-- Name: paragraph_p44_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p44_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p45_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p45_author_idx;


--
-- Name: paragraph_p45_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p45_dataset_idx;


--
-- Name: paragraph_p45_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p45_keywords_idx;


--
-- Name: paragraph_p45_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p45_outline_idx;


--
-- Name: paragraph_p45_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p45_pagenum_idx;


--
-- Name: paragraph_p45_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p45_pdate_idx;


--
-- Name: paragraph_p45_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p45_pkey;


--
-- Name: paragraph_p45_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p45_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p46_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p46_author_idx;


--
-- Name: paragraph_p46_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p46_dataset_idx;


--
-- Name: paragraph_p46_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p46_keywords_idx;


--
-- Name: paragraph_p46_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p46_outline_idx;


--
-- Name: paragraph_p46_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p46_pagenum_idx;


--
-- Name: paragraph_p46_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p46_pdate_idx;


--
-- Name: paragraph_p46_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p46_pkey;


--
-- Name: paragraph_p46_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p46_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p47_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p47_author_idx;


--
-- Name: paragraph_p47_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p47_dataset_idx;


--
-- Name: paragraph_p47_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p47_keywords_idx;


--
-- Name: paragraph_p47_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p47_outline_idx;


--
-- Name: paragraph_p47_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p47_pagenum_idx;


--
-- Name: paragraph_p47_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p47_pdate_idx;


--
-- Name: paragraph_p47_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p47_pkey;


--
-- Name: paragraph_p47_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p47_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p48_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p48_author_idx;


--
-- Name: paragraph_p48_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p48_dataset_idx;


--
-- Name: paragraph_p48_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p48_keywords_idx;


--
-- Name: paragraph_p48_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p48_outline_idx;


--
-- Name: paragraph_p48_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p48_pagenum_idx;


--
-- Name: paragraph_p48_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p48_pdate_idx;


--
-- Name: paragraph_p48_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p48_pkey;


--
-- Name: paragraph_p48_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p48_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p49_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p49_author_idx;


--
-- Name: paragraph_p49_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p49_dataset_idx;


--
-- Name: paragraph_p49_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p49_keywords_idx;


--
-- Name: paragraph_p49_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p49_outline_idx;


--
-- Name: paragraph_p49_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p49_pagenum_idx;


--
-- Name: paragraph_p49_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p49_pdate_idx;


--
-- Name: paragraph_p49_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p49_pkey;


--
-- Name: paragraph_p49_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p49_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p4_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p4_author_idx;


--
-- Name: paragraph_p4_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p4_dataset_idx;


--
-- Name: paragraph_p4_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p4_keywords_idx;


--
-- Name: paragraph_p4_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p4_outline_idx;


--
-- Name: paragraph_p4_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p4_pagenum_idx;


--
-- Name: paragraph_p4_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p4_pdate_idx;


--
-- Name: paragraph_p4_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p4_pkey;


--
-- Name: paragraph_p4_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p4_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p50_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p50_author_idx;


--
-- Name: paragraph_p50_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p50_dataset_idx;


--
-- Name: paragraph_p50_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p50_keywords_idx;


--
-- Name: paragraph_p50_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p50_outline_idx;


--
-- Name: paragraph_p50_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p50_pagenum_idx;


--
-- Name: paragraph_p50_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p50_pdate_idx;


--
-- Name: paragraph_p50_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p50_pkey;


--
-- Name: paragraph_p50_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p50_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p51_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p51_author_idx;


--
-- Name: paragraph_p51_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p51_dataset_idx;


--
-- Name: paragraph_p51_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p51_keywords_idx;


--
-- Name: paragraph_p51_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p51_outline_idx;


--
-- Name: paragraph_p51_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p51_pagenum_idx;


--
-- Name: paragraph_p51_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p51_pdate_idx;


--
-- Name: paragraph_p51_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p51_pkey;


--
-- Name: paragraph_p51_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p51_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p52_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p52_author_idx;


--
-- Name: paragraph_p52_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p52_dataset_idx;


--
-- Name: paragraph_p52_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p52_keywords_idx;


--
-- Name: paragraph_p52_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p52_outline_idx;


--
-- Name: paragraph_p52_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p52_pagenum_idx;


--
-- Name: paragraph_p52_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p52_pdate_idx;


--
-- Name: paragraph_p52_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p52_pkey;


--
-- Name: paragraph_p52_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p52_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p53_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p53_author_idx;


--
-- Name: paragraph_p53_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p53_dataset_idx;


--
-- Name: paragraph_p53_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p53_keywords_idx;


--
-- Name: paragraph_p53_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p53_outline_idx;


--
-- Name: paragraph_p53_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p53_pagenum_idx;


--
-- Name: paragraph_p53_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p53_pdate_idx;


--
-- Name: paragraph_p53_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p53_pkey;


--
-- Name: paragraph_p53_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p53_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p54_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p54_author_idx;


--
-- Name: paragraph_p54_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p54_dataset_idx;


--
-- Name: paragraph_p54_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p54_keywords_idx;


--
-- Name: paragraph_p54_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p54_outline_idx;


--
-- Name: paragraph_p54_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p54_pagenum_idx;


--
-- Name: paragraph_p54_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p54_pdate_idx;


--
-- Name: paragraph_p54_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p54_pkey;


--
-- Name: paragraph_p54_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p54_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p55_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p55_author_idx;


--
-- Name: paragraph_p55_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p55_dataset_idx;


--
-- Name: paragraph_p55_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p55_keywords_idx;


--
-- Name: paragraph_p55_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p55_outline_idx;


--
-- Name: paragraph_p55_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p55_pagenum_idx;


--
-- Name: paragraph_p55_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p55_pdate_idx;


--
-- Name: paragraph_p55_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p55_pkey;


--
-- Name: paragraph_p55_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p55_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p56_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p56_author_idx;


--
-- Name: paragraph_p56_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p56_dataset_idx;


--
-- Name: paragraph_p56_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p56_keywords_idx;


--
-- Name: paragraph_p56_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p56_outline_idx;


--
-- Name: paragraph_p56_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p56_pagenum_idx;


--
-- Name: paragraph_p56_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p56_pdate_idx;


--
-- Name: paragraph_p56_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p56_pkey;


--
-- Name: paragraph_p56_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p56_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p57_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p57_author_idx;


--
-- Name: paragraph_p57_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p57_dataset_idx;


--
-- Name: paragraph_p57_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p57_keywords_idx;


--
-- Name: paragraph_p57_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p57_outline_idx;


--
-- Name: paragraph_p57_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p57_pagenum_idx;


--
-- Name: paragraph_p57_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p57_pdate_idx;


--
-- Name: paragraph_p57_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p57_pkey;


--
-- Name: paragraph_p57_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p57_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p58_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p58_author_idx;


--
-- Name: paragraph_p58_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p58_dataset_idx;


--
-- Name: paragraph_p58_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p58_keywords_idx;


--
-- Name: paragraph_p58_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p58_outline_idx;


--
-- Name: paragraph_p58_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p58_pagenum_idx;


--
-- Name: paragraph_p58_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p58_pdate_idx;


--
-- Name: paragraph_p58_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p58_pkey;


--
-- Name: paragraph_p58_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p58_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p59_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p59_author_idx;


--
-- Name: paragraph_p59_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p59_dataset_idx;


--
-- Name: paragraph_p59_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p59_keywords_idx;


--
-- Name: paragraph_p59_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p59_outline_idx;


--
-- Name: paragraph_p59_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p59_pagenum_idx;


--
-- Name: paragraph_p59_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p59_pdate_idx;


--
-- Name: paragraph_p59_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p59_pkey;


--
-- Name: paragraph_p59_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p59_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p5_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p5_author_idx;


--
-- Name: paragraph_p5_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p5_dataset_idx;


--
-- Name: paragraph_p5_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p5_keywords_idx;


--
-- Name: paragraph_p5_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p5_outline_idx;


--
-- Name: paragraph_p5_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p5_pagenum_idx;


--
-- Name: paragraph_p5_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p5_pdate_idx;


--
-- Name: paragraph_p5_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p5_pkey;


--
-- Name: paragraph_p5_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p5_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p60_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p60_author_idx;


--
-- Name: paragraph_p60_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p60_dataset_idx;


--
-- Name: paragraph_p60_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p60_keywords_idx;


--
-- Name: paragraph_p60_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p60_outline_idx;


--
-- Name: paragraph_p60_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p60_pagenum_idx;


--
-- Name: paragraph_p60_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p60_pdate_idx;


--
-- Name: paragraph_p60_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p60_pkey;


--
-- Name: paragraph_p60_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p60_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p61_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p61_author_idx;


--
-- Name: paragraph_p61_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p61_dataset_idx;


--
-- Name: paragraph_p61_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p61_keywords_idx;


--
-- Name: paragraph_p61_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p61_outline_idx;


--
-- Name: paragraph_p61_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p61_pagenum_idx;


--
-- Name: paragraph_p61_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p61_pdate_idx;


--
-- Name: paragraph_p61_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p61_pkey;


--
-- Name: paragraph_p61_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p61_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p62_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p62_author_idx;


--
-- Name: paragraph_p62_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p62_dataset_idx;


--
-- Name: paragraph_p62_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p62_keywords_idx;


--
-- Name: paragraph_p62_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p62_outline_idx;


--
-- Name: paragraph_p62_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p62_pagenum_idx;


--
-- Name: paragraph_p62_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p62_pdate_idx;


--
-- Name: paragraph_p62_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p62_pkey;


--
-- Name: paragraph_p62_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p62_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p63_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p63_author_idx;


--
-- Name: paragraph_p63_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p63_dataset_idx;


--
-- Name: paragraph_p63_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p63_keywords_idx;


--
-- Name: paragraph_p63_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p63_outline_idx;


--
-- Name: paragraph_p63_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p63_pagenum_idx;


--
-- Name: paragraph_p63_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p63_pdate_idx;


--
-- Name: paragraph_p63_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p63_pkey;


--
-- Name: paragraph_p63_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p63_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p6_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p6_author_idx;


--
-- Name: paragraph_p6_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p6_dataset_idx;


--
-- Name: paragraph_p6_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p6_keywords_idx;


--
-- Name: paragraph_p6_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p6_outline_idx;


--
-- Name: paragraph_p6_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p6_pagenum_idx;


--
-- Name: paragraph_p6_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p6_pdate_idx;


--
-- Name: paragraph_p6_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p6_pkey;


--
-- Name: paragraph_p6_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p6_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p7_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p7_author_idx;


--
-- Name: paragraph_p7_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p7_dataset_idx;


--
-- Name: paragraph_p7_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p7_keywords_idx;


--
-- Name: paragraph_p7_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p7_outline_idx;


--
-- Name: paragraph_p7_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p7_pagenum_idx;


--
-- Name: paragraph_p7_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p7_pdate_idx;


--
-- Name: paragraph_p7_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p7_pkey;


--
-- Name: paragraph_p7_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p7_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p8_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p8_author_idx;


--
-- Name: paragraph_p8_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p8_dataset_idx;


--
-- Name: paragraph_p8_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p8_keywords_idx;


--
-- Name: paragraph_p8_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p8_outline_idx;


--
-- Name: paragraph_p8_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p8_pagenum_idx;


--
-- Name: paragraph_p8_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p8_pdate_idx;


--
-- Name: paragraph_p8_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p8_pkey;


--
-- Name: paragraph_p8_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p8_source_url_source_page_pagenum_idx;


--
-- Name: paragraph_p9_author_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_author ATTACH PARTITION public.paragraph_p9_author_idx;


--
-- Name: paragraph_p9_dataset_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.fki_dataset ATTACH PARTITION public.paragraph_p9_dataset_idx;


--
-- Name: paragraph_p9_keywords_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_keywords ATTACH PARTITION public.paragraph_p9_keywords_idx;


--
-- Name: paragraph_p9_outline_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_outline ATTACH PARTITION public.paragraph_p9_outline_idx;


--
-- Name: paragraph_p9_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pagenum ATTACH PARTITION public.paragraph_p9_pagenum_idx;


--
-- Name: paragraph_p9_pdate_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_pdate ATTACH PARTITION public.paragraph_p9_pdate_idx;


--
-- Name: paragraph_p9_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.paragraph_part_pkey ATTACH PARTITION public.paragraph_p9_pkey;


--
-- Name: paragraph_p9_source_url_source_page_pagenum_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_paragraph_source ATTACH PARTITION public.paragraph_p9_source_url_source_page_pagenum_idx;


--
-- Name: text_embeddings_p0_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p0_embedding_idx;


--
-- Name: text_embeddings_p0_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p0_pkey;


--
-- Name: text_embeddings_p10_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p10_embedding_idx;


--
-- Name: text_embeddings_p10_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p10_pkey;


--
-- Name: text_embeddings_p11_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p11_embedding_idx;


--
-- Name: text_embeddings_p11_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p11_pkey;


--
-- Name: text_embeddings_p12_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p12_embedding_idx;


--
-- Name: text_embeddings_p12_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p12_pkey;


--
-- Name: text_embeddings_p13_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p13_embedding_idx;


--
-- Name: text_embeddings_p13_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p13_pkey;


--
-- Name: text_embeddings_p14_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p14_embedding_idx;


--
-- Name: text_embeddings_p14_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p14_pkey;


--
-- Name: text_embeddings_p15_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p15_embedding_idx;


--
-- Name: text_embeddings_p15_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p15_pkey;


--
-- Name: text_embeddings_p16_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p16_embedding_idx;


--
-- Name: text_embeddings_p16_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p16_pkey;


--
-- Name: text_embeddings_p17_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p17_embedding_idx;


--
-- Name: text_embeddings_p17_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p17_pkey;


--
-- Name: text_embeddings_p18_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p18_embedding_idx;


--
-- Name: text_embeddings_p18_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p18_pkey;


--
-- Name: text_embeddings_p19_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p19_embedding_idx;


--
-- Name: text_embeddings_p19_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p19_pkey;


--
-- Name: text_embeddings_p1_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p1_embedding_idx;


--
-- Name: text_embeddings_p1_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p1_pkey;


--
-- Name: text_embeddings_p20_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p20_embedding_idx;


--
-- Name: text_embeddings_p20_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p20_pkey;


--
-- Name: text_embeddings_p21_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p21_embedding_idx;


--
-- Name: text_embeddings_p21_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p21_pkey;


--
-- Name: text_embeddings_p22_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p22_embedding_idx;


--
-- Name: text_embeddings_p22_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p22_pkey;


--
-- Name: text_embeddings_p23_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p23_embedding_idx;


--
-- Name: text_embeddings_p23_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p23_pkey;


--
-- Name: text_embeddings_p24_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p24_embedding_idx;


--
-- Name: text_embeddings_p24_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p24_pkey;


--
-- Name: text_embeddings_p25_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p25_embedding_idx;


--
-- Name: text_embeddings_p25_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p25_pkey;


--
-- Name: text_embeddings_p26_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p26_embedding_idx;


--
-- Name: text_embeddings_p26_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p26_pkey;


--
-- Name: text_embeddings_p27_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p27_embedding_idx;


--
-- Name: text_embeddings_p27_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p27_pkey;


--
-- Name: text_embeddings_p28_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p28_embedding_idx;


--
-- Name: text_embeddings_p28_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p28_pkey;


--
-- Name: text_embeddings_p29_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p29_embedding_idx;


--
-- Name: text_embeddings_p29_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p29_pkey;


--
-- Name: text_embeddings_p2_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p2_embedding_idx;


--
-- Name: text_embeddings_p2_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p2_pkey;


--
-- Name: text_embeddings_p30_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p30_embedding_idx;


--
-- Name: text_embeddings_p30_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p30_pkey;


--
-- Name: text_embeddings_p31_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p31_embedding_idx;


--
-- Name: text_embeddings_p31_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p31_pkey;


--
-- Name: text_embeddings_p32_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p32_embedding_idx;


--
-- Name: text_embeddings_p32_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p32_pkey;


--
-- Name: text_embeddings_p33_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p33_embedding_idx;


--
-- Name: text_embeddings_p33_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p33_pkey;


--
-- Name: text_embeddings_p34_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p34_embedding_idx;


--
-- Name: text_embeddings_p34_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p34_pkey;


--
-- Name: text_embeddings_p35_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p35_embedding_idx;


--
-- Name: text_embeddings_p35_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p35_pkey;


--
-- Name: text_embeddings_p36_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p36_embedding_idx;


--
-- Name: text_embeddings_p36_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p36_pkey;


--
-- Name: text_embeddings_p37_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p37_embedding_idx;


--
-- Name: text_embeddings_p37_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p37_pkey;


--
-- Name: text_embeddings_p38_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p38_embedding_idx;


--
-- Name: text_embeddings_p38_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p38_pkey;


--
-- Name: text_embeddings_p39_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p39_embedding_idx;


--
-- Name: text_embeddings_p39_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p39_pkey;


--
-- Name: text_embeddings_p3_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p3_embedding_idx;


--
-- Name: text_embeddings_p3_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p3_pkey;


--
-- Name: text_embeddings_p40_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p40_embedding_idx;


--
-- Name: text_embeddings_p40_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p40_pkey;


--
-- Name: text_embeddings_p41_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p41_embedding_idx;


--
-- Name: text_embeddings_p41_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p41_pkey;


--
-- Name: text_embeddings_p42_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p42_embedding_idx;


--
-- Name: text_embeddings_p42_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p42_pkey;


--
-- Name: text_embeddings_p43_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p43_embedding_idx;


--
-- Name: text_embeddings_p43_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p43_pkey;


--
-- Name: text_embeddings_p44_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p44_embedding_idx;


--
-- Name: text_embeddings_p44_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p44_pkey;


--
-- Name: text_embeddings_p45_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p45_embedding_idx;


--
-- Name: text_embeddings_p45_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p45_pkey;


--
-- Name: text_embeddings_p46_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p46_embedding_idx;


--
-- Name: text_embeddings_p46_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p46_pkey;


--
-- Name: text_embeddings_p47_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p47_embedding_idx;


--
-- Name: text_embeddings_p47_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p47_pkey;


--
-- Name: text_embeddings_p48_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p48_embedding_idx;


--
-- Name: text_embeddings_p48_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p48_pkey;


--
-- Name: text_embeddings_p49_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p49_embedding_idx;


--
-- Name: text_embeddings_p49_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p49_pkey;


--
-- Name: text_embeddings_p4_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p4_embedding_idx;


--
-- Name: text_embeddings_p4_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p4_pkey;


--
-- Name: text_embeddings_p50_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p50_embedding_idx;


--
-- Name: text_embeddings_p50_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p50_pkey;


--
-- Name: text_embeddings_p51_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p51_embedding_idx;


--
-- Name: text_embeddings_p51_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p51_pkey;


--
-- Name: text_embeddings_p52_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p52_embedding_idx;


--
-- Name: text_embeddings_p52_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p52_pkey;


--
-- Name: text_embeddings_p53_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p53_embedding_idx;


--
-- Name: text_embeddings_p53_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p53_pkey;


--
-- Name: text_embeddings_p54_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p54_embedding_idx;


--
-- Name: text_embeddings_p54_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p54_pkey;


--
-- Name: text_embeddings_p55_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p55_embedding_idx;


--
-- Name: text_embeddings_p55_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p55_pkey;


--
-- Name: text_embeddings_p56_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p56_embedding_idx;


--
-- Name: text_embeddings_p56_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p56_pkey;


--
-- Name: text_embeddings_p57_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p57_embedding_idx;


--
-- Name: text_embeddings_p57_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p57_pkey;


--
-- Name: text_embeddings_p58_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p58_embedding_idx;


--
-- Name: text_embeddings_p58_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p58_pkey;


--
-- Name: text_embeddings_p59_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p59_embedding_idx;


--
-- Name: text_embeddings_p59_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p59_pkey;


--
-- Name: text_embeddings_p5_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p5_embedding_idx;


--
-- Name: text_embeddings_p5_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p5_pkey;


--
-- Name: text_embeddings_p60_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p60_embedding_idx;


--
-- Name: text_embeddings_p60_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p60_pkey;


--
-- Name: text_embeddings_p61_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p61_embedding_idx;


--
-- Name: text_embeddings_p61_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p61_pkey;


--
-- Name: text_embeddings_p62_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p62_embedding_idx;


--
-- Name: text_embeddings_p62_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p62_pkey;


--
-- Name: text_embeddings_p63_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p63_embedding_idx;


--
-- Name: text_embeddings_p63_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p63_pkey;


--
-- Name: text_embeddings_p6_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p6_embedding_idx;


--
-- Name: text_embeddings_p6_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p6_pkey;


--
-- Name: text_embeddings_p7_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p7_embedding_idx;


--
-- Name: text_embeddings_p7_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p7_pkey;


--
-- Name: text_embeddings_p8_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p8_embedding_idx;


--
-- Name: text_embeddings_p8_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p8_pkey;


--
-- Name: text_embeddings_p9_embedding_idx; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.idx_text_embeddings_vchord ATTACH PARTITION public.text_embeddings_p9_embedding_idx;


--
-- Name: text_embeddings_p9_pkey; Type: INDEX ATTACH; Schema: public; Owner: myuser
--

ALTER INDEX public.text_embeddings_part_pkey ATTACH PARTITION public.text_embeddings_p9_pkey;


--
-- Name: paragraph trg_after_paragraph_insert; Type: TRIGGER; Schema: public; Owner: myuser
--

CREATE TRIGGER trg_after_paragraph_insert AFTER INSERT ON public.paragraph FOR EACH ROW EXECUTE FUNCTION public.fn_enqueue_paragraph_embedding();


--
-- Name: text_embeddings trg_after_text_embedding_insert; Type: TRIGGER; Schema: public; Owner: myuser
--

CREATE TRIGGER trg_after_text_embedding_insert AFTER INSERT ON public.text_embeddings FOR EACH ROW EXECUTE FUNCTION public.fn_dequeue_on_embedding_insert();


--
-- Name: history fk_history_user; Type: FK CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.history
    ADD CONSTRAINT fk_history_user FOREIGN KEY (user_id) REFERENCES public.user_info(id) ON DELETE CASCADE;


--
-- Name: paragraph fk_paragraph_dataset; Type: FK CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE public.paragraph
    ADD CONSTRAINT fk_paragraph_dataset FOREIGN KEY (dataset) REFERENCES public.dataset(id);


--
-- Name: text_embeddings fk_paragraph_part; Type: FK CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE public.text_embeddings
    ADD CONSTRAINT fk_paragraph_part FOREIGN KEY (id, dataset) REFERENCES public.paragraph(id, dataset) ON UPDATE CASCADE ON DELETE CASCADE;


--
-- Name: task_dbo fk_user_id; Type: FK CONSTRAINT; Schema: public; Owner: myuser
--

ALTER TABLE ONLY public.task_dbo
    ADD CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES public.user_info(id) NOT VALID;


--
-- PostgreSQL database dump complete
--

\unrestrict cUiQTb26uqSlxRkGrCdcVe6Y7H8gV9CZa5V4qeiQrJuuqKdBBVDG0kN7LcGevtU

