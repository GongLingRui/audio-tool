# 前后端对接方案

## 1. 对接概述

### 1.1 通信架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (React + Vite)                     │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Pages     │  │ Components  │  │      Stores         │ │
│  │             │  │             │  │  (Zustand)          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────────────┘ │
│         │                │                 │                 │
│         └────────────────┼─────────────────┘                 │
│                          │                                   │
│                    ┌─────┴─────┐                             │
│                    │ Services  │                             │
│                    │ (API层)   │                             │
│                    └─────┬─────┘                             │
└──────────────────────────┼───────────────────────────────────┘
                           │ HTTP/REST
                           │ WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      后端 (FastAPI)                          │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  API Routes │  │  Services   │  │      Models         │ │
│  │  (REST)     │  │ (业务逻辑)  │  │   (SQLAlchemy)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────────────┘ │
│         │                │                 │                 │
└─────────┼────────────────┼─────────────────┼────────────────┘
          │                │                 │
          ▼                ▼                 ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │   Web    │     │   LLM    │     │   TTS    │
    │  Client  │     │  Service │     │  Engine  │
    └──────────┘     └──────────┘     └──────────┘
```

### 1.2 技术对接

| 层级 | 前端 | 后端 | 通信方式 |
|------|------|------|----------|
| 路由 | React Router | FastAPI 路由 | REST API |
| 状态 | Zustand | SQLAlchemy | HTTP/JSON |
| 实时 | WebSocket | WebSocket | WS |
| 文件 | FormData | UploadFile | Multipart |
| 认证 | JWT Header | JWT Bearer | Authorization |

## 2. API 客户端实现

### 2.1 基础 API 客户端

```typescript
// src/services/api.ts
import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';

export interface ApiConfig {
  baseURL: string;
  timeout?: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

export class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor(config: ApiConfig) {
    this.client = axios.create({
      baseURL: config.baseURL,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response.data,
      (error: AxiosError<any>) => {
        if (error.response?.status === 401) {
          // Token 过期，清除登录状态
          this.token = null;
          window.location.href = '/login';
        }
        return Promise.reject(error.response?.data || error);
      }
    );
  }

  setToken(token: string) {
    this.token = token;
  }

  clearToken() {
    this.token = null;
  }

  async get<T = any>(url: string, params?: any): Promise<ApiResponse<T>> {
    return this.client.get(url, { params });
  }

  async post<T = any>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.client.post(url, data);
  }

  async put<T = any>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.client.put(url, data);
  }

  async patch<T = any>(url: string, data?: any): Promise<ApiResponse<T>> {
    return this.client.patch(url, data);
  }

  async delete<T = any>(url: string): Promise<ApiResponse<T>> {
    return this.client.delete(url);
  }

  async upload<T = any>(url: string, formData: FormData): Promise<ApiResponse<T>> {
    return this.client.post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async download(url: string, filename?: string): Promise<void> {
    const response = await this.client.get(url, {
      responseType: 'blob',
    });

    const blob = new Blob([response.data]);
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename || 'download';
    link.click();
    URL.revokeObjectURL(link.href);
  }
}

// 创建全局 API 客户端实例
export const apiClient = new ApiClient({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 60000, // 音频生成需要更长的超时时间
});
```

### 2.2 书籍 API 服务

```typescript
// src/services/books.ts
import { apiClient, ApiResponse } from './api';
import { Book, BookCreate, BookUpdate } from '@/types/book';

export interface BookListParams {
  page?: number;
  page_size?: number;
  search?: string;
  sort?: string;
  order?: 'asc' | 'desc';
}

export interface BookListResponse {
  items: Book[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface BookContentResponse {
  content: string;
  chapters: Array<{
    index: number;
    title: string;
    offset: number;
  }>;
  metadata: {
    title: string;
    author: string;
    total_chars: number;
  };
}

export const booksApi = {
  // 获取书籍列表
  list: (params: BookListParams = {}): Promise<ApiResponse<BookListResponse>> => {
    return apiClient.get('/books', params);
  },

  // 获取书籍详情
  get: (bookId: string): Promise<ApiResponse<Book>> => {
    return apiClient.get(`/books/${bookId}`);
  },

  // 上传书籍
  upload: (file: File, metadata?: Partial<BookCreate>): Promise<ApiResponse<Book>> => {
    const formData = new FormData();
    formData.append('file', file);

    if (metadata?.title) {
      formData.append('title', metadata.title);
    }
    if (metadata?.author) {
      formData.append('author', metadata.author);
    }
    if (metadata?.cover) {
      formData.append('cover', metadata.cover);
    }

    return apiClient.upload('/books/upload', formData);
  },

  // 获取书籍内容
  getContent: (
    bookId: string,
    format: 'plain' | 'markdown' | 'html' = 'plain',
    chapter?: number
  ): Promise<ApiResponse<BookContentResponse>> => {
    return apiClient.get(`/books/${bookId}/content`, { format, chapter });
  },

  // 更新书籍
  update: (bookId: string, data: BookUpdate): Promise<ApiResponse<Book>> => {
    return apiClient.patch(`/books/${bookId}`, data);
  },

  // 删除书籍
  delete: (bookId: string): Promise<ApiResponse<{ deleted: boolean }>> => {
    return apiClient.delete(`/books/${bookId}`);
  },
};
```

### 2.3 项目 API 服务

```typescript
// src/services/projects.ts
import { apiClient, ApiResponse } from './api';
import { Project, ProjectCreate, ProjectUpdate } from '@/types/project';

export interface ProjectListParams {
  book_id?: string;
  status?: string;
  page?: number;
  page_size?: number;
}

export interface ProjectListResponse {
  items: Project[];
  total: number;
  page: number;
  page_size: number;
}

export interface ProjectProgress {
  total_chunks: number;
  completed_chunks: number;
  percentage: number;
}

export const projectsApi = {
  // 获取项目列表
  list: (params: ProjectListParams = {}): Promise<ApiResponse<ProjectListResponse>> => {
    return apiClient.get('/projects', params);
  },

  // 获取项目详情
  get: (projectId: string): Promise<ApiResponse<Project>> => {
    return apiClient.get(`/projects/${projectId}`);
  },

  // 创建项目
  create: (data: ProjectCreate): Promise<ApiResponse<Project>> => {
    return apiClient.post('/projects', data);
  },

  // 更新项目
  update: (projectId: string, data: ProjectUpdate): Promise<ApiResponse<Project>> => {
    return apiClient.patch(`/projects/${projectId}`, data);
  },

  // 删除项目
  delete: (projectId: string): Promise<ApiResponse<{ deleted: boolean }>> => {
    return apiClient.delete(`/projects/${projectId}`);
  },

  // 获取生成进度
  getProgress: (projectId: string): Promise<ApiResponse<ProjectProgress>> => {
    return apiClient.get(`/projects/${projectId}/chunks/progress`);
  },

  // 下载音频
  downloadAudio: (projectId: string, format: 'mp3' | 'wav' | 'zip' = 'mp3'): Promise<void> => {
    return apiClient.download(`/projects/${projectId}/audio/download?format=${format}`);
  },
};
```

### 2.4 脚本 API 服务

```typescript
// src/services/scripts.ts
import { apiClient, ApiResponse } from './api';
import { ScriptEntry } from '@/types/script';

export interface ScriptGenerateOptions {
  system_prompt?: string;
  user_prompt?: string;
  options?: {
    max_chunk_size?: number;
    detect_emotions?: boolean;
    detect_sections?: boolean;
  };
}

export const scriptsApi = {
  // 生成脚本
  generate: (
    projectId: string,
    options: ScriptGenerateOptions = {}
  ): Promise<ApiResponse<{ script_id: string; status: string }>> => {
    return apiClient.post(`/projects/${projectId}/scripts/generate`, options);
  },

  // 获取脚本状态
  getStatus: (projectId: string): Promise<ApiResponse<any>> => {
    return apiClient.get(`/projects/${projectId}/scripts/status`);
  },

  // 获取脚本内容
  get: (projectId: string): Promise<ApiResponse<{ content: ScriptEntry[] }>> => {
    return apiClient.get(`/projects/${projectId}/scripts`);
  },

  // 更新脚本
  update: (
    projectId: string,
    content: ScriptEntry[]
  ): Promise<ApiResponse<any>> => {
    return apiClient.patch(`/projects/${projectId}/scripts`, { content });
  },

  // 审查脚本
  review: (
    projectId: string,
    options: {
      auto_fix?: boolean;
      check_rules?: {
        speaker_consistency?: boolean;
        text_continuity?: boolean;
        emotion_accuracy?: boolean;
      };
    } = {}
  ): Promise<ApiResponse<any>> => {
    return apiClient.post(`/projects/${projectId}/scripts/review`, options);
  },

  // 批准脚本
  approve: (projectId: string): Promise<ApiResponse<any>> => {
    return apiClient.post(`/projects/${projectId}/scripts/approve`);
  },
};
```

### 2.5 音频 API 服务

```typescript
// src/services/audio.ts
import { apiClient, ApiResponse } from './api';
import { Chunk, ChunkUpdate } from '@/types/audio';

export const audioApi = {
  // 获取音频块列表
  getChunks: (
    projectId: string,
    params: {
      speaker?: string;
      status?: string;
      page?: number;
      page_size?: number;
    } = {}
  ): Promise<ApiResponse<{ items: Chunk[]; total: number }>> => {
    return apiClient.get(`/projects/${projectId}/chunks`, params);
  },

  // 生成单个音频块
  generateChunk: (projectId: string, chunkId: string): Promise<ApiResponse<any>> => {
    return apiClient.post(`/projects/${projectId}/chunks/${chunkId}/generate`);
  },

  // 批量生成音频
  generateBatch: (
    projectId: string,
    chunkIds: string[],
    mode: 'parallel' | 'sequential' = 'parallel',
    workers: number = 2
  ): Promise<ApiResponse<{ task_id: string }>> => {
    return apiClient.post(`/projects/${projectId}/chunks/generate-batch`, {
      chunk_ids: chunkIds,
      mode,
      workers,
    });
  },

  // 快速批量生成
  generateFast: (projectId: string): Promise<ApiResponse<{ task_id: string }>> => {
    return apiClient.post(`/projects/${projectId}/chunks/generate-fast`);
  },

  // 更新音频块
  updateChunk: (projectId: string, chunkId: string, data: ChunkUpdate): Promise<ApiResponse<Chunk>> => {
    return apiClient.patch(`/projects/${projectId}/chunks/${chunkId}`, data);
  },

  // 重新生成音频块
  regenerateChunk: (projectId: string, chunkId: string): Promise<ApiResponse<any>> => {
    return apiClient.post(`/projects/${projectId}/chunks/${chunkId}/regenerate`);
  },

  // 合并音频
  mergeAudio: (
    projectId: string,
    options: {
      pause_between_speakers?: number;
      pause_same_speaker?: number;
      output_format?: 'mp3' | 'wav';
      bitrate?: string;
    } = {}
  ): Promise<ApiResponse<any>> => {
    return apiClient.post(`/projects/${projectId}/audio/merge`, options);
  },

  // 获取合并后的音频
  getAudio: (projectId: string): Promise<ApiResponse<{
    audio_url: string;
    duration: number;
    file_size: number;
  }>> => {
    return apiClient.get(`/projects/${projectId}/audio`);
  },
};
```

### 2.6 语音 API 服务

```typescript
// src/services/voices.ts
import { apiClient, ApiResponse } from './api';

export interface Voice {
  id: string;
  name: string;
  gender?: 'male' | 'female';
  language?: string;
}

export interface VoiceConfig {
  speaker: string;
  voice_type: 'custom' | 'clone' | 'lora' | 'design';
  voice_name?: string;
  style?: string;
  ref_audio_path?: string;
  lora_model_path?: string;
  language?: string;
}

export const voicesApi = {
  // 获取语音列表
  list: (): Promise<ApiResponse<{
    custom: Voice[];
    lora: Voice[];
  }>> => {
    return apiClient.get('/voices');
  },

  // 获取项目语音配置
  getProjectVoices: (projectId: string): Promise<ApiResponse<{ voices: VoiceConfig[] }>> => {
    return apiClient.get(`/projects/${projectId}/voices`);
  },

  // 解析脚本发言人
  parseSpeakers: (projectId: string): Promise<ApiResponse<{
    speakers: string[];
    total_entries: number;
  }>> => {
    return apiClient.post(`/projects/${projectId}/voices/parse`);
  },

  // 设置语音配置
  setVoiceConfig: (
    projectId: string,
    voices: VoiceConfig[]
  ): Promise<ApiResponse<{ updated: boolean; count: number }>> => {
    return apiClient.post(`/projects/${projectId}/voices/config`, { voices });
  },

  // 预览语音
  preview: (data: {
    text: string;
    voice_type: string;
    voice_name?: string;
    instruct?: string;
  }): Promise<ApiResponse<{ audio_url: string; duration: number }>> => {
    return apiClient.post('/voices/preview', data);
  },

  // 上传参考音频
  uploadCloneAudio: (audio: File, text: string): Promise<ApiResponse<{
    audio_path: string;
    duration: number;
  }>> => {
    const formData = new FormData();
    formData.append('audio', audio);
    formData.append('text', text);
    return apiClient.upload('/voices/clone/upload', formData);
  },

  // 语音设计
  designVoice: (data: {
    description: string;
    gender?: 'male' | 'female';
    age_range?: string;
    style?: string;
  }): Promise<ApiResponse<{ preview_url: string; voice_id: string }>> => {
    return apiClient.post('/voices/design', data);
  },
};
```

### 2.7 笔记 API 服务

```typescript
// src/services/highlights.ts
import { apiClient, ApiResponse } from './api';
import { Highlight, HighlightCreate, Note } from '@/types/note';

export const highlightsApi = {
  // 获取高亮列表
  list: (
    bookId: string,
    params: {
      color?: string;
      chapter?: string;
      has_note?: boolean;
    } = {}
  ): Promise<ApiResponse<{ items: Highlight[]; total: number }>> => {
    return apiClient.get(`/books/${bookId}/highlights`, params);
  },

  // 创建高亮
  create: (bookId: string, data: HighlightCreate): Promise<ApiResponse<Highlight>> => {
    return apiClient.post(`/books/${bookId}/highlights`, data);
  },

  // 更新高亮
  update: (
    highlightId: string,
    data: Partial<Highlight>
  ): Promise<ApiResponse<Highlight>> => {
    return apiClient.patch(`/highlights/${highlightId}`, data);
  },

  // 删除高亮
  delete: (highlightId: string): Promise<ApiResponse<{ deleted: boolean }>> => {
    return apiClient.delete(`/highlights/${highlightId}`);
  },

  // 添加/更新笔记
  setNote: (highlightId: string, content: string): Promise<ApiResponse<Note>> => {
    return apiClient.put(`/highlights/${highlightId}/note`, { content });
  },

  // 删除笔记
  deleteNote: (highlightId: string): Promise<ApiResponse<{ deleted: boolean }>> => {
    return apiClient.delete(`/highlights/${highlightId}/note`);
  },

  // 导出笔记
  exportNotes: (
    bookId: string,
    format: 'json' | 'markdown' | 'csv' = 'markdown'
  ): Promise<void> => {
    return apiClient.download(`/books/${bookId}/notes/export?format=${format}`);
  },
};
```

## 3. React Query 集成

### 3.1 Query Client 配置

```typescript
// src/lib/query-client.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 分钟
    },
  },
});
```

### 3.2 自定义 Hooks

```typescript
// src/hooks/useBooks.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { booksApi, BookListParams } from '@/services/books';
import { Book, BookCreate, BookUpdate } from '@/types/book';

export function useBooks(params: BookListParams = {}) {
  return useQuery({
    queryKey: ['books', params],
    queryFn: () => booksApi.list(params).then((res) => res.data!),
  });
}

export function useBook(bookId: string) {
  return useQuery({
    queryKey: ['book', bookId],
    queryFn: () => booksApi.get(bookId).then((res) => res.data!),
    enabled: !!bookId,
  });
}

export function useBookContent(bookId: string, format: 'plain' | 'markdown' | 'html' = 'plain') {
  return useQuery({
    queryKey: ['book-content', bookId, format],
    queryFn: () => booksApi.getContent(bookId, format).then((res) => res.data!),
    enabled: !!bookId,
  });
}

export function useUploadBook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ file, metadata }: { file: File; metadata?: Partial<BookCreate> }) =>
      booksApi.upload(file, metadata).then((res) => res.data!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['books'] });
    },
  });
}

export function useUpdateBook(bookId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: BookUpdate) =>
      booksApi.update(bookId, data).then((res) => res.data!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['book', bookId] });
    },
  });
}

export function useDeleteBook() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (bookId: string) =>
      booksApi.delete(bookId).then((res) => res.data!),
    onSuccess: (_, bookId) => {
      queryClient.invalidateQueries({ queryKey: ['books'] });
      queryClient.removeQueries({ queryKey: ['book', bookId] });
    },
  });
}
```

```typescript
// src/hooks/useProject.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { projectsApi, scriptsApi, audioApi } from '@/services';
import { Project, ProjectCreate, ProjectUpdate } from '@/types/project';

export function useProjects(bookId?: string) {
  return useQuery({
    queryKey: ['projects', { bookId }],
    queryFn: () => projectsApi.list({ book_id: projectId }).then((res) => res.data!),
    enabled: !!bookId,
  });
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: ['project', projectId],
    queryFn: () => projectsApi.get(projectId).then((res) => res.data!),
    enabled: !!projectId,
    refetchInterval: (data) => {
      // 如果项目正在处理，每 5 秒刷新一次
      return data?.state.data?.status === 'processing' ? 5000 : false;
    },
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ProjectCreate) =>
      projectsApi.create(data).then((res) => res.data!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
}

export function useScriptGeneration(projectId: string) {
  return useMutation({
    mutationFn: (options: any) =>
      scriptsApi.generate(projectId, options).then((res) => res.data!),
  });
}

export function useAudioGeneration(projectId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (chunkIds?: string[]) =>
      chunkIds
        ? audioApi.generateBatch(projectId, chunkIds)
        : audioApi.generateFast(projectId).then((res) => res.data!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectId] });
    },
  });
}
```

## 4. 状态管理对接

### 4.1 更新 BookStore

```typescript
// src/stores/bookStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { booksApi, BookListParams } from '@/services/books';

interface Book {
  id: string;
  title: string;
  author?: string;
  cover_url?: string;
  file_type: string;
  total_pages?: number;
  total_chars?: number;
  progress: number;
  created_at: string;
  updated_at: string;
}

interface BookStore {
  books: Book[];
  currentBook: Book | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchBooks: (params?: BookListParams) => Promise<void>;
  fetchBook: (bookId: string) => Promise<void>;
  setCurrentBook: (book: Book | null) => void;
  addBook: (book: Book) => void;
  removeBook: (bookId: string) => void;
  updateBookProgress: (bookId: string, progress: number) => void;
}

export const useBookStore = create<BookStore>()(
  persist(
    (set, get) => ({
      books: [],
      currentBook: null,
      isLoading: false,
      error: null,

      fetchBooks: async (params) => {
        set({ isLoading: true, error: null });
        try {
          const response = await booksApi.list(params);
          if (response.success && response.data) {
            set({ books: response.data.items, isLoading: false });
          }
        } catch (error: any) {
          set({ error: error.message || '获取书籍列表失败', isLoading: false });
        }
      },

      fetchBook: async (bookId) => {
        set({ isLoading: true, error: null });
        try {
          const response = await booksApi.get(bookId);
          if (response.success && response.data) {
            set({ currentBook: response.data, isLoading: false });
          }
        } catch (error: any) {
          set({ error: error.message || '获取书籍详情失败', isLoading: false });
        }
      },

      setCurrentBook: (book) => {
        set({ currentBook: book });
      },

      addBook: (book) => {
        set((state) => ({ books: [book, ...state.books] }));
      },

      removeBook: (bookId) => {
        set((state) => ({
          books: state.books.filter((b) => b.id !== bookId),
          currentBook: state.currentBook?.id === bookId ? null : state.currentBook,
        }));
      },

      updateBookProgress: (bookId, progress) => {
        set((state) => ({
          books: state.books.map((b) =>
            b.id === bookId ? { ...b, progress } : b
          ),
          currentBook:
            state.currentBook?.id === bookId
              ? { ...state.currentBook, progress }
              : state.currentBook,
        }));
      },
    }),
    {
      name: 'book-storage',
      partialize: (state) => ({
        books: state.books,
        currentBook: state.currentBook?.id
          ? { id: state.currentBook.id }
          : null,
      }),
    }
  )
);
```

### 4.2 创建 ProjectStore

```typescript
// src/stores/projectStore.ts
import { create } from 'zustand';
import { projectsApi, scriptsApi, audioApi } from '@/services';

interface Chunk {
  id: string;
  speaker: string;
  text: string;
  instruct?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  audio_path?: string;
  duration?: number;
  order_index: number;
}

interface ScriptEntry {
  index: number;
  speaker: string;
  text: string;
  instruct?: string;
  emotion?: string;
  section?: string;
}

interface ProjectStore {
  // Project state
  currentProject: any | null;
  script: ScriptEntry[] | null;
  chunks: Chunk[];
  voiceConfigs: any[];
  isGenerating: boolean;
  progress: {
    total: number;
    completed: number;
    percentage: number;
  } | null;

  // Actions
  setCurrentProject: (project: any) => void;
  fetchProject: (projectId: string) => Promise<void>;
  generateScript: (options?: any) => Promise<void>;
  fetchScript: () => Promise<void>;
  updateScript: (script: ScriptEntry[]) => Promise<void>;
  fetchChunks: () => Promise<void>;
  generateAudio: (chunkIds?: string[]) => Promise<void>;
  mergeAudio: () => Promise<void>;
  reset: () => void;
}

export const useProjectStore = create<ProjectStore>((set, get) => ({
  currentProject: null,
  script: null,
  chunks: [],
  voiceConfigs: [],
  isGenerating: false,
  progress: null,

  setCurrentProject: (project) => {
    set({ currentProject: project });
  },

  fetchProject: async (projectId) => {
    try {
      const response = await projectsApi.get(projectId);
      if (response.success && response.data) {
        set({ currentProject: response.data });
      }
    } catch (error) {
      console.error('Failed to fetch project:', error);
    }
  },

  generateScript: async (options = {}) => {
    const { currentProject } = get();
    if (!currentProject) return;

    set({ isGenerating: true });
    try {
      await scriptsApi.generate(currentProject.id, options);
      // 轮询状态
      // ...
    } catch (error) {
      console.error('Failed to generate script:', error);
    } finally {
      set({ isGenerating: false });
    }
  },

  fetchScript: async () => {
    const { currentProject } = get();
    if (!currentProject) return;

    try {
      const response = await scriptsApi.get(currentProject.id);
      if (response.success && response.data) {
        set({ script: response.data.content });
      }
    } catch (error) {
      console.error('Failed to fetch script:', error);
    }
  },

  updateScript: async (script) => {
    const { currentProject } = get();
    if (!currentProject) return;

    try {
      await scriptsApi.update(currentProject.id, script);
      set({ script });
    } catch (error) {
      console.error('Failed to update script:', error);
    }
  },

  fetchChunks: async () => {
    const { currentProject } = get();
    if (!currentProject) return;

    try {
      const response = await audioApi.getChunks(currentProject.id);
      if (response.success && response.data) {
        set({ chunks: response.data.items });
      }
    } catch (error) {
      console.error('Failed to fetch chunks:', error);
    }
  },

  generateAudio: async (chunkIds) => {
    const { currentProject } = get();
    if (!currentProject) return;

    set({ isGenerating: true });
    try {
      if (chunkIds) {
        await audioApi.generateBatch(currentProject.id, chunkIds);
      } else {
        await audioApi.generateFast(currentProject.id);
      }
    } catch (error) {
      console.error('Failed to generate audio:', error);
    } finally {
      set({ isGenerating: false });
    }
  },

  mergeAudio: async () => {
    const { currentProject } = get();
    if (!currentProject) return;

    try {
      await audioApi.mergeAudio(currentProject.id);
    } catch (error) {
      console.error('Failed to merge audio:', error);
    }
  },

  reset: () => {
    set({
      currentProject: null,
      script: null,
      chunks: [],
      voiceConfigs: [],
      isGenerating: false,
      progress: null,
    });
  },
}));
```

## 5. WebSocket 集成

### 5.1 WebSocket Hook

```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  data: any;
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket connected');
      reconnectAttempts.current = 0;
      options.onConnect?.();
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        options.onMessage?.(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      options.onDisconnect?.();

      // 尝试重连
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log(`Reconnecting... (attempt ${reconnectAttempts.current})`);
          connect();
        }, delay);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      options.onError?.(error);
    };

    wsRef.current = ws;
  }, [url, options]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    sendMessage,
    disconnect,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  };
}
```

### 5.2 项目进度订阅

```typescript
// src/hooks/useProjectProgress.ts
import { useWebSocket } from './useWebSocket';
import { useEffect } from 'react';
import { useProjectStore } from '@/stores/projectStore';

export function useProjectProgress(projectId: string) {
  const { progress, chunks } = useProjectStore();

  const handleMessage = (message: any) => {
    if (message.type === 'audio_progress') {
      const { completed, total, percentage } = message.data.progress;
      // 更新进度
      useProjectStore.getState().setProgress({
        total,
        completed,
        percentage,
      });

      // 更新 chunk 状态
      if (message.data.chunk_id) {
        useProjectStore.getState().updateChunkStatus(
          message.data.chunk_id,
          message.data.status
        );
      }
    }
  };

  const ws = useWebSocket(`ws://localhost:8000/ws/${projectId}`, {
    onMessage: handleMessage,
  });

  useEffect(() => {
    if (ws.isConnected) {
      ws.sendMessage({
        type: 'subscribe',
        channel: 'audio_progress',
      });
    }
  }, [ws.isConnected]);

  return progress;
}
```

## 6. 类型定义

### 6.1 通用类型

```typescript
// src/types/common.ts
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}
```

### 6.2 书籍类型

```typescript
// src/types/book.ts
export interface Book {
  id: string;
  user_id: string;
  title: string;
  author?: string;
  cover_url?: string;
  file_type: 'txt' | 'pdf' | 'epub';
  total_pages?: number;
  total_chars?: number;
  progress: number;
  created_at: string;
  updated_at: string;
}

export interface BookCreate {
  title?: string;
  author?: string;
  cover?: File;
}

export interface BookUpdate {
  title?: string;
  author?: string;
  cover_url?: string;
  progress?: number;
}
```

### 6.3 项目类型

```typescript
// src/types/project.ts
export interface Project {
  id: string;
  book_id: string;
  book_title: string;
  name: string;
  description?: string;
  status: 'draft' | 'processing' | 'completed' | 'failed';
  config: ProjectConfig;
  audio_path?: string;
  duration?: number;
  created_at: string;
  updated_at: string;
  progress?: {
    total_chunks: number;
    completed_chunks: number;
    percentage: number;
  };
}

export interface ProjectConfig {
  tts_mode: 'local' | 'external';
  tts_url?: string;
  language: string;
  parallel_workers?: number;
}

export interface ProjectCreate {
  book_id: string;
  name: string;
  description?: string;
  config?: ProjectConfig;
}

export interface ProjectUpdate {
  name?: string;
  description?: string;
  config?: Partial<ProjectConfig>;
}
```

### 6.4 脚本类型

```typescript
// src/types/script.ts
export interface ScriptEntry {
  index: number;
  speaker: string;
  text: string;
  instruct?: string;
  emotion?: string;
  section?: string;
}
```

### 6.5 音频类型

```typescript
// src/types/audio.ts
export interface Chunk {
  id: string;
  project_id: string;
  speaker: string;
  text: string;
  instruct?: string;
  emotion?: string;
  section?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  audio_path?: string;
  duration?: number;
  order_index: number;
}

export interface ChunkUpdate {
  text?: string;
  instruct?: string;
  speaker?: string;
}
```

### 6.6 笔记类型

```typescript
// src/types/note.ts
export interface Highlight {
  id: string;
  user_id: string;
  book_id: string;
  chunk_id?: string;
  text: string;
  color: 'yellow' | 'green' | 'blue' | 'pink';
  start_offset: number;
  end_offset: number;
  chapter?: string;
  note?: Note;
  created_at: string;
}

export interface HighlightCreate {
  text: string;
  color: 'yellow' | 'green' | 'blue' | 'pink';
  start_offset: number;
  end_offset: number;
  chapter?: string;
  chunk_id?: string;
  note?: string;
}

export interface Note {
  id: string;
  highlight_id: string;
  content: string;
  created_at: string;
  updated_at: string;
}
```

## 7. 环境配置

### 7.1 环境变量

```bash
# .env.development
VITE_API_BASE_URL=http://localhost:8000/api
VITE_WS_BASE_URL=ws://localhost:8000/ws
```

```bash
# .env.production
VITE_API_BASE_URL=https://api.example.com/api
VITE_WS_BASE_URL=wss://api.example.com/ws
```

### 7.2 Vite 配置

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```
