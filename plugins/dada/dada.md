# Database for Digital Arts (DADA) 后台 API

## 数据类型

### `Dada` 数据项

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|_id|str|数据库内的 ID|
|author|str|作者/艺术家名称|
|content|str|简介/网页主要内容|
|lang|str|内容简介所用语言|
|images|List[MediaItem]|多媒体信息列表|
|keywords|List[str]|关键词列表，供全文索引用|
|tags|List[str]|标签列表|
|pdate|Union[str, datetime]|日期或年份（四位数）或yyyy-mm格式的年月信息|
|source|SourceObject|来源信息|
|src|str|图片来源的绝对路径|

### `MediaItem` 多媒体内容

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|_id|str|数据库内的 ID|
|width|int|宽度，0表示未检测，一般为0|
|height|int|高度，0表示未检测，一般为0|
|item_type|str|多媒体内容类型，图像为 image，视频为 video，音频为 audio|
|thumbnail|str|缩略图地址，一般为空|
|source|SourceObject|来源信息|
|src|str|图片来源的绝对路径|

### `SourceObject` 来源信息

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|file|str|本地文件路径，可选|
|page|int|本地文件为 PDF 时的页码，从 0 开始，可选|
|url|str|网址路径，可选|

- `file` 和 `url` 至少有一项有内容。
- `page` 仅在 `file` 表示的是一个 PDF 文件时存在。
- （8/30更新）注：使用 `MediaItem` 或 `Dada` 的 `src` 字段获得完整路径。如果为空说明无对应图片。


### `APIUpdate` 更新返回

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|success|bool|操作成功与否|
|bundle|Any|附加数据|


### `APIResults` 查询返回

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|query|str|查询字符串，若不是查询所得，则为空字符串|
|results|List[Dada]|查询结果|
|total|int|查询总数，-1表示不确定（可以通过 `results` 的长度确定）|


## API 调用

如无特殊说明，请求体（body）应为 JSON 格式传递参数，均以 POST 方式提交。注意设置 `Content-Type: application/json`。

### 登录

入口为 `/api/authenticate`。

| 字段名 | 类型 | 意义 |
|-------|-----|------|
|username|str|用户名|
|password|str|密码|

返回一个 `APIUpdate`，其中 `bundle` 为 `{ "token": "<TOKEN>" }`。请注意保存 `token`，并在之后的调用中，总是在 HTTP 头的 `X-Authentication-Token` 中写入 Token，如：

```
headers: {
    "X-Authentication-Token": _token
}
```

测试期间，不做鉴权要求，所有 `/api/dada` 下的 API 均可直接访问。（8/30更新：做一个简单的登录界面即可。）


### 增删改查操作

增删改查操作的入口为 `/api/dada`。HTTP Method 决定了采取何种操作。

- 新增

使用 PUT 方法新增项目。PUT 的请求体为一个 `Dada` 数据项，可以不写全字段，不要设置 `_id` 字段。
返回一个 `APIUpdate`，其中 `bundle` 为新增的完整 `Dada` 数据项。

- 删除

使用 DELETE 方法删除项目。此时调用的路径格式为：`/api/dada/${_id}`，请求体为空。
返回一个 `APIUpdate`，其中 `bundle` 为被删除项的 ID。

- 修改单个项目

使用 POST 方法修改单个项目。此时调用的路径格式为：`/api/dada/${_id}`，请求体为修改后的 `Dada` 数据项。可以只传递需要修改的字段，如 `{"content": "New Content"}`。
此时，请求体中的 `_id` 将被抛弃，也不能修改项目的 ID。
返回一个 `APIUpdate`，其中 `bundle` 为仅包含修改了的字段和值的 `Partial[Dada]` 数据项。

- 修改多个项目，每个项目不同的字段和值

使用 POST 方法修改多个项目，并赋予每个项目不同的值。此时调用的路径格式为：`/api/dada`，请求体为一个 `Dict[str, Partial[Dada]]`，即一个 ID 和对应要修改的字段与新值。
返回一个 `APIUpdate`，其中 `bundle` 为一个 `Dict[str, Partial[Dada]]`，意义与请求体相同。

- 修改多个项目，每个项目相同的字段和值

使用 POST 方法修改多个项目，并赋予每个项目不同的值。此时调用的路径格式为：`/api/dada`，请求体是在 `Partial[Dada]` 的基础上增加一个 `ids` 字段，其为一个 `List[str]`，给出所有要更新的项目的 ID。
返回一个 `APIUpdate`，其中 `bundle` 为一个 `Dict[str, Partial[Dada]]`，意义与每项目不同字段和值时的请求体相同。

- 查询，按照表达式查询

使用 GET 方法进行查询，参数如下：

| 参数名 | 类型 | 意义 |
|-------|-----|------|
|query|str|查询表达式，可以认为默认只对全文关键字 keywords 字段进行匹配|
|limit|int|返回结果数量限制，默认为0即无限制|
|offset|int|要跳过的结果数量，默认为0即从头开始|
|sort|str|排序，默认为按照ID升序|

返回一个 `APIResults`，即查询结果。排序时，用 `-` 表示逆序，如 `-pdate` 表示按日期逆序。多个字段排序可以用 `,` 组合起来，如 `_id,-pdate` 表示先按 ID 升序，再按日期降序。

- 按 ID 获取单个项目

使用 GET 方法，路径格式为：`/api/dada/${_id}`。不要带任何参数。

返回一个 `Dada` 数据项。

### 抓取网页

入口：`/api/dada/fetch`

参数：

| 参数名 | 类型 | 意义 |
|-------|-----|------|
|url|str|要抓取的网址，必须|
|可选参数：|
|depth|str|抓取深度，默认为1|
|assignments|Dict[str, str]|字段名和元素、属性对照表（注）|
|selector|str|指定结果所在的 css selector，默认为 body 标签|
|scopes|List[str]|允许的网址范围，即检查 URL 是否以这些字符串打头|
|tags|List[str]|额外添加的标签|
|detail_link|str|检验是否需要抓取详情的正则表达式|
|list_link|str|检验是否需要抓取链接的正则表达式|
|img_pattern|str|图像检索标记，一般留空|

注：`assignments`的格式写作 `<field>="<css selector>//<attribute>"`，多个之间用 `,` 分割。如 `title="div.some-class a[href]//text"`，表示获取 `some-class` 类的 `div` 中，具有 `href` 属性的 `a` 标签的 `text` 属性即文本，保存到 `title` 字段中。

返回一个 `APIResults`，即抓取结果。如果抓取深度比较大或链接数量比较多，很可能运行非常久的时间，但这个先不管它，测试时务必使用 `depth` 为 1 的情况。

### 调用语言模型（GPT）

入口：`/api/dada/llm`

参数：`messages`，格式与 GPT-3.5 的 `messages` 相同。
返回一个 `APIResults`，其中数据项的 `content` 由 GPT 生成的内容填充，而其他字段均为空。

注：该接口不稳定，可能调用失败。

### 调用语言模型时的提示语

入口：`/api/dada/prompts`

用 POST 方法，参数如下：

| 参数名 | 类型 | 意义 |
|-------|-----|------|
|action|str|新建（create）或删除（delete），为空时表示查询|
|prompt|str|要新建或删除的提示语|

当 `action` 非空时，返回一个 `APIUpdate`，其中 `bundle` 为提示语内容。否则返回一个 `APIResults`，其中 `results` 列出了所有可选的提示语。


### 生成全文关键字索引

入口：`/api/dada/keywords`

参数：`ids`，要生成关键字索引的数据项 ID 列表。
返回一个 `APIUpdate`，其中 `bundle` 为空。
