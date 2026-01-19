import hashlib
from typing import Type
import os
from flask import Response, request, send_file
from flask_restful import Resource, reqparse
from sqlalchemy.dialects.postgresql import insert

from .app import (
    UUID,
    Resource,
    ResponseTuple,
    api,
    app,
    assert_admin,
    config,
    storage,
    oidc,
    request,
)
from .models import (
    Base,
    Dataset,
    History,
    Paragraph,
    TaskDBO,
    Terms,
    TextEmbeddings,
    UserInfo,
    db_session,
    is_uuid_literal,
    redis_auto_renew_cache
)
from .worker import add_task, delete_task, get_task_result, get_task_stats, clear_tasks


def paginate_cache_key(_, stmt, get_results=True, get_total=True):
    if get_total and not get_results:
        if hasattr(stmt, 'statement'): stmt = stmt.statement
        stmt_str = str(stmt) + str(sorted(stmt.compile().params.items()))
        return hashlib.sha1(stmt_str.encode('utf-8')).hexdigest()
    return ''
    

class JindaiResource(Resource):

    def __init__(self, model_cls: Type[Base]) -> None:
        super().__init__()
        self.model = model_cls

    @redis_auto_renew_cache(cache_key=paginate_cache_key)
    def paginate(self, stmt, get_results=True, get_total=True):
        data = {"results": [], "total": -1}
        offset, limit = int(request.args.get("offset", "0")), int(
            request.args.get("limit", "100")
        )
        if get_total:
            data["total"] = stmt.count()
        if get_results:
            stmt = stmt.offset(offset).limit(limit)
            data["results"] = [r.as_dict() for r in stmt]
        return data

    def get_object_by_id(self, resource_id):
        return db_session.query(self.model).get(UUID(resource_id))

    def get(self, resource_id=None):
        if resource_id:
            result = self.get_object_by_id(resource_id)
            if not result:
                return {
                    "error": f"No resource of {self.model.__name__} with id {resource_id}"
                }, 404
            return result.as_dict()
        else:
            results = db_session.query(self.model)
            return self.paginate(results), 200

    def post(self, resource_id=""):
        data = request.json
        new_item = self.model(**data)

        try:
            self._on_create_or_update(new_item)
            db_session.add(new_item)
            db_session.commit()
            return new_item.as_dict(), 201
        except Exception as e:
            db_session.rollback()
            return {"error": str(e)}, 500

    def put(self, resource_id=""):
        data = request.json
        item = self.get_object_by_id(resource_id)
        if not item:
            return {
                "error": f"No existing resource of {self.model.__name__} with id {data.get('id')}"
            }, 404
        data.pop("id", "")
        for k, v in data.items():
            setattr(item, k, v)
        self._on_create_or_update(item)
        try:
            db_session.commit()
            return item.as_dict(), 200
        except Exception as e:
            db_session.rollback()
            return {"error": str(e)}, 500

    def delete(self, resource_id):
        item = self.get_object_by_id(resource_id)
        if not item:
            return {
                "error": f"No existing resource of {self.model.__name__} with id {resource_id}"
            }, 404

        try:
            db_session.delete(item)
            db_session.commit()
            return {"message": "Deletion succeeded"}, 200
        except Exception as e:
            db_session.rollback()
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

    def get_object_by_id(self, resource_id):
        if not is_uuid_literal(resource_id):
            return Dataset.get(resource_id, True)
        return super().get_object_by_id(resource_id)

    def get(self, resource_id=None):

        if not resource_id:
            hierarchy = Dataset.get_hierarchy()
            return {"results": hierarchy}, 200

        return super().get(resource_id)

    def post(self, resource_id):
        assert_admin()
        return super().post(resource_id)

    def delete(self, resource_id):
        assert_admin()
        return super().delete(resource_id)

    def put(self, resource_id=""):
        assert_admin()
        ds = self.get_object_by_id(resource_id)
        if ds is None:
            return "Not Found", 404
        return ds.rename_dataset(request.json["name"])


class TaskDBOResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(TaskDBO)


class ParagraphResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(Paragraph)

    def post(self):
        data = request.json
        if data.get("search"):
            query = Paragraph.build_query(data)
            return (
                self.paginate(
                    query, get_results="total" not in data, get_total="total" in data
                ),
                200,
            )
        else:
            return super().post()

    def _on_create_or_update(self, item: Paragraph):
        if item.keywords:
            db_session.execute(
                insert(Terms)
                .values([{"term": kw} for kw in item.keywords])
                .on_conflict_do_nothing()
            )
            db_session.commit()


class TextEmbeddingsResource(JindaiResource):
    def __init__(self) -> None:
        super().__init__(TextEmbeddings)

    def post(self, resource_id=""):
        paragraph = db_session.query(Paragraph).get(resource_id)
        if paragraph:
            embedding = TextEmbeddings.get_embedding(paragraph.content)
            te = self.get_object_by_id(resource_id)
            if te:
                te.embedding = embedding
            else:
                te = TextEmbeddings(id=resource_id, embedding=embedding)
                db_session.add(te)
            db_session.commit()
        return {"id": resource_id}, 201

    def put(self, resource_id=""):
        return self.post(resource_id)[0], 200

    def get(self):
        return (
            db_session.query(TextEmbeddings)
            .filter(TextEmbeddings.chunk_id == 1)
            .count()
        )


class OIDCUserInfoResource(Resource):

    def get(self, section):
        if oidc.user_loggedin:
            if section:
                info = {}
                if section == "histories":
                    pass
                elif section == "":
                    pass
                return info, 200
            else:
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
            return {
                "error": "User not logged in",
                "redirect": oidc.redirect_to_auth_server(request.url).location,
            }, 401


class WorkerResource(Resource):
    def get(self, task_id=""):
        if not task_id:
            return get_task_stats()
        else:
            return get_task_result(task_id)

    def post(self):
        add_task(**request.json)

    def delete(self, task_id=""):
        if task_id in ["", "pending", "processing", "completed", "failed"]:
            clear_tasks(task_id)
        else:
            delete_task(task_id)
        return True


class FileManagerResource(Resource):
    def get(self, file_path: str = "") -> ResponseTuple | Response:
        """
        GET /api/files/[file_path]
        - 无file_path：列出根目录文件
        - file_path是目录：列出该目录下的文件/文件夹
        - file_path是文件：下载该文件
        - metadata参数：仅返回文件/目录元信息，不下载文件
        - page参数：下载文件指定页码（仅对PDF有效），从0开始
        """
        parser = reqparse.RequestParser()
        parser.add_argument("metadata", type=bool, default=False, location="args")
        parser.add_argument("page", type=int, location="args")
        args = parser.parse_args()

        try:
            target_path = storage.safe_join(file_path)
        except ValueError as e:
            return {"error": str(e)}, 403

        # 路径不存在
        if not os.path.exists(target_path):
            return {"error": "文件/目录不存在"}, 404

        # 是目录：返回目录列表
        if os.path.isdir(target_path):
            dir_data = storage.list_directory(
                target_path, only_basic=not args["metadata"]
            )
            return dir_data, 200

        # 是文件：返回元数据
        if args["metadata"]:
            file_info = storage.get_file_info(target_path)
            return file_info, 200

        # 是文件：下载文件
        try:
            buf, mime_type, download_name = storage.read_file(file_path, args["page"])
            return send_file(
                buf,
                mimetype=mime_type,
                as_attachment=False,
                download_name=download_name,
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            return {"error": f"文件下载失败：{str(e)}"}, 500

    def post(self, file_path="") -> ResponseTuple:
        """
        POST /api/files/[file_path]
        - file_path为空：上传文件到根目录
        - file_path是目录：上传文件到该目录
        - 支持创建空目录（通过参数 is_directory=true）
        """
        # 解析请求参数
        args = {"is_directory": False, "name": ""}
        if request.is_json:
            args.update(request.json)

        # 创建空目录逻辑
        if args["is_directory"] and args["name"]:
            try:
                dir_info = storage.create_directory(file_path, args["name"])
                return dir_info, 201
            except ValueError as e:
                return {"error": str(e)}, 409
            except Exception as e:
                return {"error": f"创建目录失败：{str(e)}"}, 500

        # 文件上传逻辑
        if "file" not in request.files:
            return {"error": "未找到上传的文件"}, 400
        file = request.files["file"]
        if not file.filename:
            return {"error": "文件名不能为空"}, 400

        try:
            file_info = storage.save_file(file, file_path)
            return file_info, 201
        except ValueError as e:
            return {"error": str(e)}, 403
        except Exception as e:
            return {"error": f"文件上传失败：{str(e)}"}, 500

    def put(self, file_path) -> ResponseTuple:
        """
        PUT /api/files/[file_path]
        - 重命名文件/目录
        - 移动文件/目录
        - 联动数据库更新Paragraph表的source_url字段
        """
        data = request.get_json()
        if not data or not data.get("name") and not data.get("path"):
            return {"error": "需要提供 name 或 path 参数"}, 400

        try:
            move_data = storage.move_or_rename(
                file_path, data.get("name"), data.get("path")
            )
            old_rel_path = move_data["old_relative_path"]
            new_rel_path = move_data["new_info"]["relative_path"]
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            return {"error": f"操作失败：{str(e)}"}, 500

        # 数据库联动更新 - 原逻辑完整保留
        if os.path.isfile(storage.safe_join(old_rel_path)):
            pattern = "%" + old_rel_path
        else:
            pattern = old_rel_path + "/%"

        db_session.query(Paragraph).filter(Paragraph.source_url.like(pattern)).update(
            {
                Paragraph.source_url: db.func.replace(
                    Paragraph.source_url, old_rel_path, new_rel_path
                )
            },
            synchronize_session=False,
        )
        db_session.commit()

        return move_data["new_info"], 200

    def delete(self, file_path: str) -> ResponseTuple:
        """
        DELETE /api/files/[file_path]
        - 删除文件/空目录
        - 管理员权限校验 + 原逻辑完整保留
        """
        assert_admin()
        try:
            storage.delete(file_path)
            return {"message": "文件/目录删除成功"}, 200
        except ValueError as e:
            return {"error": str(e)}, 400
        except Exception as e:
            return {"error": f"删除失败：{str(e)}"}, 500


def apply_resources(api):
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
    api.add_resource(OIDCUserInfoResource, "/api/user", "/api/user/<string:section>")
    api.add_resource(WorkerResource, "/api/worker", "/api/worker/<string:task_id>")
    api.add_resource(
        FileManagerResource, "/api/files", "/api/files/", "/api/files/<path:file_path>"
    )


@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()  # 关键：防止连接泄露
