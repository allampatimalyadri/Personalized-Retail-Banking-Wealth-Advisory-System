import re
import os
import psycopg

from dotenv import load_dotenv
from psycopg.rows import dict_row
from langchain_core.tools import tool

from app.core.db import get_vector_store

load_dotenv()

COLLECTION_NAME="hr_support_desk"

_row_conn=(
    os.getenv(
        "PG_CONNECTION_STRING"
    ).replace(
        "postgresql+psycopg",
        "postgresql"
    )
)

###################################################
# MODE DETECTION
###################################################

_KEYWORD_PATTERNS=[

    r"[A-Z]{2,}-\d{4}-\w+",

    r"\b[A-Z]{2,5}\b",

    r"\d{6,}"

]

_KEYWORD_RE=re.compile(
    "|".join(
        _KEYWORD_PATTERNS
    )
)


def detect_mode_core(
        query:str
)->str:

    query=query.strip()

    if _KEYWORD_RE.search(
            query
    ):
        return "keyword"

    if len(
            query.split()
    )<=3:
        return "hybrid"

    return "vector"


###################################################
# VECTOR
###################################################

def vector_search_core(
        query:str,
        k:int=5
)->list[str]:

    vector_store=get_vector_store(
        collection_name=
        COLLECTION_NAME
    )

    docs=vector_store.similarity_search(
        query,
        k=k
    )

    return [

        d.page_content

        for d in docs

    ]


###################################################
# FTS
###################################################

def fts_search_core(
        query:str,
        k:int=5
)->list[str]:

    sql="""

    SELECT

    e.document AS content

    FROM langchain_pg_embedding e

    JOIN langchain_pg_collection c

    ON c.uuid=e.collection_id

    WHERE

    c.name=%(collection)s

    AND

    to_tsvector(
    'english',
    e.document
    )

    @@

    plainto_tsquery(
    'english',
    %(query)s
    )

    ORDER BY
    ts_rank(
        to_tsvector(
        'english',
        e.document
        ),
        plainto_tsquery(
        'english',
        %(query)s
        )
    ) DESC

    LIMIT %(k)s

    """

    with psycopg.connect(
            _row_conn,
            row_factory=dict_row
    ) as conn:

        with conn.cursor() as cur:

            cur.execute(

                sql,

                {

                    "query":query,

                    "collection":
                    COLLECTION_NAME,

                    "k":k

                }

            )

            rows=cur.fetchall()

    return [

        x["content"]

        for x in rows

    ]


###################################################
# HYBRID (RRF)
###################################################

def hybrid_search_core(
        query:str,
        k:int=5
)->list[str]:

    vector_docs=vector_search_core(
        query,
        k
    )

    keyword_docs=fts_search_core(
        query,
        k
    )

    scores={}
    doc_map={}

    for rank,doc in enumerate(
            vector_docs
    ):

        key=doc[:120]

        scores[key]=(
            scores.get(
                key,
                0
            )

            +

            1/(60+rank+1)
        )

        doc_map[key]=doc


    for rank,doc in enumerate(
            keyword_docs
    ):

        key=doc[:120]

        scores[key]=(

            scores.get(
                key,
                0
            )

            +

            1/(60+rank+1)

        )

        doc_map[key]=doc


    ranked=sorted(

        scores.items(),

        key=lambda x:x[1],

        reverse=True
    )

    return [

        doc_map[key]

        for key,_ in ranked[:k]

    ]


###################################################
# TOOLS
###################################################

@tool
def detect_mode(
        query:str
)->str:
    """
    Detect retrieval strategy:
    keyword/hybrid/vector
    """

    return detect_mode_core(
        query
    )


@tool
def query_documents(
        query:str,
        k:int=5
)->str:
    """
    Main RAG retrieval tool
    """

    mode=detect_mode_core(
        query
    )

    if mode=="keyword":

        docs=fts_search_core(
            query,
            k
        )

    elif mode=="hybrid":

        docs=hybrid_search_core(
            query,
            k
        )

    else:

        docs=vector_search_core(
            query,
            k
        )

    if not docs:

        return "No information found"

    return "\n\n".join(
        docs
    )


@tool
def fts_search(
        query:str
)->str:
    """
    keyword search
    """

    return "\n\n".join(

        fts_search_core(
            query
        )

    )


@tool
def hybrid_search(
        query:str
)->str:
    """
    hybrid search
    """

    return "\n\n".join(

        hybrid_search_core(
            query
        )

    )