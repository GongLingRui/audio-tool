# Read-Rhyme 测试指南

本文档提供了Read-Rhyme项目的详细测试指南，包括如何运行测试、编写测试以及调试测试问题。

## 目录

- [测试概览](#测试概览)
- [后端测试](#后端测试)
- [前端测试](#前端测试)
- [测试覆盖率](#测试覆盖率)
- [调试测试](#调试测试)

## 测试概览

项目包含完整的测试套件：

| 测试类型 | 文件位置 | 测试数量 | 状态 |
|---------|---------|---------|------|
| 后端API测试 (完整) | `backend/tests/test_api.py` | 13+ | ✅ 通过 |
| 后端API测试 (简化) | `backend/tests/test_api_simple.py` | 13 | ✅ 通过 |
| 前端服务测试 | `read-rhyme/src/test/services.test.ts` | 22 | ✅ 通过 |

## 后端测试

### 环境设置

1. 确保已安装所有依赖：
```bash
cd backend
pip install -r requirements.txt
```

2. 安装测试依赖：
```bash
pip install pytest pytest-asyncio httpx
```

### 运行测试

#### 运行所有测试

```bash
# 运行所有测试
pytest tests/

# 输出详细信息
pytest tests/ -v -s

# 显示详细错误信息
pytest tests/ -v -s --tb=short
```

#### 运行简化测试（推荐）

简化测试不需要数据库，运行更快：

```bash
pytest tests/test_api_simple.py -v -s
```

#### 运行特定测试类

```bash
# 只测试语音API
pytest tests/test_api_simple.py::TestVoicesAPI -v -s

# 只测试RAG API
pytest tests/test_api_simple.py::TestRAGAPI -v -s
```

#### 运行特定测试方法

```bash
# 只测试语音列表
pytest tests/test_api_simple.py::TestVoicesAPI::test_list_voices -v -s
```

### 后端测试结构

```python
class TestVoicesAPI:
    """测试语音管理API"""

    async def test_list_voices(self, client):
        """测试列出可用语音"""
        response = await client.get("/api/voices")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    async def test_voice_reference(self, client):
        """测试获取语音参考词汇"""
        response = await client.get("/api/voices/reference")
        assert response.status_code == 200
```

### 测试覆盖的功能

#### 1. 语音API (TestVoicesAPI)
- ✅ 列出可用语音
- ✅ 获取语音参考词汇
- ✅ 从描述设计语音

#### 2. 语音样式API (TestVoiceStylingAPI)
- ✅ 列出情感预设
- ✅ 列出支持的语言

#### 3. RAG API (TestRAGAPI)
- ✅ 文档索引
- ✅ 查询文档
- ✅ 获取统计信息

#### 4. 音频工具API (TestAudioToolsAPI)
- ✅ 获取音频质量指南

#### 5. 情感预设API (TestEmotionPresetsAPI)
- ✅ 列出所有情感预设
- ✅ 按类别获取预设

#### 6. LoRA训练API (TestLoRATrainingAPI)
- ✅ 获取训练要求
- ✅ 获取配置模板

### 后端测试最佳实践

1. **使用AsyncClient**: 所有测试应该使用AsyncClient
2. **验证响应格式**: 检查 `response.status_code` 和 `data` 字段
3. **使用断言**: 使用 `assert` 验证结果
4. **添加打印**: 使用 `print()` 调试测试

## 前端测试

### 环境设置

1. 安装依赖：
```bash
cd read-rhyme
npm install
```

### 运行测试

```bash
# 运行所有测试
npm test

# 运行测试并监听变化
npm run test:watch

# 运行测试并生成覆盖率报告
npm test -- --coverage
```

### 前端测试结构

```typescript
describe("ApiClient", () => {
  describe("HTTP Methods", () => {
    it("should make GET request", async () => {
      const mockResponse = { success: true, data: { result: "test" } };
      mockAxiosInstance.get.mockResolvedValue(mockResponse);

      const result = await apiClient.get("/test");
      expect(result).toEqual(mockResponse);
    });
  });
});
```

### 测试覆盖的功能

#### 1. API客户端测试
- ✅ Token管理
- ✅ HTTP方法 (GET, POST, PUT, PATCH, DELETE)
- ✅ 文件上传
- ✅ 响应接口

#### 2. 服务集成测试
- ✅ 语音服务 (voicesApi)
- ✅ 语音样式服务 (voiceStylingApi)
- ✅ RAG服务 (ragApi)
- ✅ 认证服务 (authService)
- ✅ 书籍服务 (booksApi)
- ✅ 项目服务 (projectsApi)
- ✅ 脚本服务 (scriptsApi)
- ✅ 音频服务 (audioApi)
- ✅ WebSocket服务 (websocketService)

### 前端测试最佳实践

1. **使用vitest**: 项目使用vitest作为测试框架
2. **Mock外部依赖**: 使用vi.mock()模拟axios等外部依赖
3. **异步测试**: 使用async/await处理异步操作
4. **清晰的测试描述**: 使用有意义的测试名称

## 测试覆盖率

### 后端覆盖率

运行以下命令生成覆盖率报告：

```bash
pytest tests/ --cov=app --cov-report=html
```

覆盖率报告将生成在 `htmlcov/index.html`

### 前端覆盖率

```bash
npm test -- --coverage
```

## 调试测试

### 后端测试调试

1. **启用详细输出**:
```bash
pytest tests/ -v -s --tb=long
```

2. **使用pdb调试器**:
```python
def test_example():
    import pdb; pdb.set_trace()
    # 测试代码
```

3. **查看日志**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 常见后端测试问题

#### 问题1: 模块导入错误

```
ModuleNotFoundError: No module named 'app'
```

**解决方案**: 确保从项目根目录运行测试，或将项目路径添加到Python路径：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### 问题2: 数据库连接错误

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
```

**解决方案**: 使用简化测试版本或确保数据库已初始化：

```bash
alembic upgrade head
```

#### 问题3: API响应格式不匹配

```
AssertionError: assert 'data' in data
```

**解决方案**: 确保API响应使用ApiResponse包装：

```python
return ApiResponse(data=result)
```

### 前端测试调试

1. **启用详细输出**:
```bash
npm test -- --reporter=verbose
```

2. **使用console.log**:
```typescript
it("should work", () => {
  console.log("Debug info");
  expect(true).toBe(true);
});
```

3. **仅运行失败的测试**:
```bash
npm test -- --run
```

### 常见前端测试问题

#### 问题1: Mock不工作

```
TypeError: Cannot read properties of undefined
```

**解决方案**: 确保在测试文件顶部正确mock了依赖：

```typescript
vi.mock("axios", () => ({
  default: {
    create: vi.fn(),
  },
}));
```

#### 问题2: 异步测试超时

**解决方案**: 增加测试超时时间：

```typescript
it("should complete", async () => {
  // 测试代码
}, 10000); // 10秒超时
```

## 持续集成

### GitHub Actions示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd read-rhyme
          npm install
      - name: Run tests
        run: |
          cd read-rhyme
          npm test
```

## 贡献测试

### 添加新的后端测试

1. 在 `backend/tests/` 中创建或编辑测试文件
2. 继承测试类模式
3. 使用async/await
4. 添加适当的断言
5. 运行测试验证

### 添加新的前端测试

1. 在 `read-rhyme/src/test/` 中创建测试文件
2. 使用vitest API
3. Mock外部依赖
4. 添加清晰的测试描述
5. 运行测试验证

## 测试检查清单

在提交代码前，确保：

- [ ] 所有后端测试通过
- [ ] 所有前端测试通过
- [ ] 新功能有相应的测试
- [ ] 测试覆盖率没有下降
- [ ] 测试文档已更新

## 资源

- [pytest文档](https://docs.pytest.org/)
- [vitest文档](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
