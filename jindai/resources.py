from typing import Type

import jieba
from sqlalchemy import or_
from sqlalchemy.sql import func, text
from sqlalchemy.dialects.postgresql import insert

from .app import UUID, Resource, api, db, request, oidc, app
from .models import (
    Base,
    Dataset,
    History,
    Paragraph,
    TaskDBO,
    TextEmbeddings,
    UserInfo,
    Terms,
)


class JindaiResource(Resource):

    def __init__(self, model_cls: Type[Base]) -> None:
        super().__init__()
        self.model = model_cls

    def paginate(self, results):
        stmt = results
        # stmt = results.add_columns(func.count(1).over().label("total_count"))
        offset, limit = int(request.args.get("offset", "0")), int(
            request.args.get("limit", "100")
        )
        results = stmt.offset(offset).limit(limit)
        data = {"results": [r.as_dict() for r in results], "total": 0}
        if data["results"]:
            data["total"] = stmt.count()
        return data

    def get(self, resource_id=None):
        if resource_id:
            result = self.model.query.get(UUID(resource_id))
            if not result:
                return {
                    "error": f"No resource of {self.model.__name__} with id {resource_id}"
                }, 404
            return result.as_dict()
        else:
            results = self.model.query
            return {"results": self.paginate(results), "count": results.count()}, 200

    def post(self):
        data = request.json
        new_item = self.model(**data)

        try:
            self._on_create_or_update(new_item)
            db.session.add(new_item)
            db.session.commit()
            return new_item.as_dict(), 201
        except Exception as e:
            db.session.rollback()
            return {"error": str(e)}, 500

    def put(self):
        data = request.json
        item = self.model.query.get(id=UUID(data.get("id")))
        if not item:
            return {
                "error": f"No existing resource of {self.model.__name__} with id {data.get('id')}"
            }, 404
        del data["id"]
        for k, v in data.item():
            setattr(item, k, v)
        self._on_create_or_update(item)
        try:
            db.session.commit()
            return item.as_dict(), 200
        except Exception as e:
            db.session.rollback()
            return {"error": str(e)}, 500

    def delete(self, resource_id):
        item = self.model.query.get(UUID(resource_id))
        if not item:
            return {
                "error": f"No existing resource of {self.model.__name__} with id {resource_id}"
            }, 404

        try:
            db.session.delete(item)
            db.session.commit()
            return {"message": "Deletion succeeded"}, 200
        except Exception as e:
            db.session.rollback()
            return {"error": str(e)}, 500

    def _on_create_or_update(self, item):
        pass


class UserInfoResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(UserInfo)


class HistoryResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(History)


class DatasetResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(Dataset)
        
    def get(self, resource_id=None):
        
        def _dataset_sort_key(ds: Dataset):
            return len(ds.name.split('--')), ds.order_weight, ds.name
        
        if not resource_id:
            datasets = Dataset.query.all()
            sorted_datasets = sorted(datasets, key=_dataset_sort_key)
            hierarchy = []
            for dataset in sorted_datasets:
                current_level = hierarchy
                parts = dataset.name.split('--')
                for parti, part in enumerate(parts):
                    found = False
                    for item in current_level:
                        if item["title"] == part:
                            current_level = item.setdefault("children", [])
                            found = True
                            break
                    if not found:
                        new_item = {"title": part, "children": [], "value": '--'.join(parts[:parti+1])}
                        current_level.append(new_item)
                        current_level = new_item["children"]
            return {"results": hierarchy, "count": len(datasets)}, 200
        
        return super().get(resource_id)


class TaskDBOResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(TaskDBO)


class ParagraphResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(Paragraph)
        
    def get_embedding(self, text: str):
        raise NotImplementedError("Embedding service not implemented")

    def post(self):
        data = request.json
        if search := data.get("search"):
            if search.startswith('?'):
                param = text(search[1:])
            elif search.startswith('*'):
                param = Paragraph.content.ilike(f"%{search.strip('*')}%")
            elif search.startswith(':'):
                embedding = self.get_embedding(search.strip(':'))
                param = TextEmbeddings.embedding.op('<=>')(embedding) < 0.3
            else:
                param = Paragraph.keywords.contains(
                    [_.strip().lower() for _ in jieba.cut(search) if _.strip()]
                )
            if datasets := data.get("datasets"):
                dataset_filters = [Dataset.name.in_(datasets)]
                for dataset_prefix in datasets:
                    dataset_filters.append(Dataset.name.ilike(f"{dataset_prefix}--%"))
                param &= Paragraph.dataset.in_(Dataset.query.filter(or_(*dataset_filters)).with_entities(Dataset.id))
            if sources := data.get("sources"):
                param &= Paragraph.source_url.in_(sources)
            query = Paragraph.query.filter(param)
            return self.paginate(query), 200
        else:
            return super().post()

    def _on_create_or_update(self, item: Paragraph):
        if item.keywords:
            db.session.execute(
                insert(Terms)
                .values([{"term": kw} for kw in item.keywords])
                .on_conflict_do_nothing()
            )
            db.session.commit()


class TextEmbeddingsResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(TextEmbeddings)
        
        
class OIDCUserInfoResource(Resource):
    def get(self):
        if oidc.user_loggedin:
            info = oidc.user_getinfo(
                [
                    "preferred_username",
                    "email",
                    "given_name",
                    "family_name",
                    "sub",
                ]
            )
            return info, 200
        else:
            return {"error": "User not logged in", "redirect": oidc.redirect_to_auth_server(request.url).location}, 401


def apply_resources():
    api.add_resource(
        DatasetResource, "/api/datasets", "/api/datasets/<string:resource_id>"
    )
    api.add_resource(UserInfoResource, "/api/users", "/api/users/<string:resource_id>")
    api.add_resource(
        HistoryResource, "/api/histories", "/api/histories/<string:resource_id>"
    )
    api.add_resource(TaskDBOResource, "/api/tasks", "/api/tasks/<string:resource_id>")
    api.add_resource(
        ParagraphResource, "/api/paragraphs", "/api/paragraphs/<string:resource_id>"
    )
    api.add_resource(
        TextEmbeddingsResource,
        "/api/embeddings",
        "/api/embeddings/<string:resource_id>",
    )
    api.add_resource(OIDCUserInfoResource, "/api/user")
