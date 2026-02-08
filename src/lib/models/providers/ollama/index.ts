import { UIConfigField } from '@/lib/config/types';
import { getConfiguredModelProviderById } from '@/lib/config/serverRegistry';
import BaseModelProvider from '../../base/provider';
import { Model, ModelList, ProviderMetadata } from '../../types';
import BaseLLM from '../../base/llm';
import BaseEmbedding from '../../base/embedding';
import OllamaLLM from './ollamaLLM';
import OllamaEmbedding from './ollamaEmbedding';

interface OllamaConfig {
  mode: 'local' | 'cloud';
  baseURL?: string;
  apiKey?: string;
}

const providerConfigFields: UIConfigField[] = [
  {
    type: 'select',
    name: 'Mode',
    key: 'mode',
    description: 'Choose between Local Ollama or Ollama Cloud',
    required: true,
    default: 'local',
    options: [
      { name: 'Local (Self-hosted)', value: 'local' },
      { name: 'Cloud (ollama.com)', value: 'cloud' },
    ],
    scope: 'server',
  },
  {
    type: 'string',
    name: 'Base URL',
    key: 'baseURL',
    description: 'Only required for Local mode',
    required: false,
    placeholder: process.env.DOCKER
      ? 'http://host.docker.internal:11434'
      : 'http://localhost:11434',
    env: 'OLLAMA_BASE_URL',
    scope: 'server',
  },
  {
    type: 'string',
    name: 'API Key',
    key: 'apiKey',
    description: 'Required for Cloud mode. Get one at ollama.com',
    required: false,
    placeholder: 'ollama_xxxxxxxxxxxxx',
    env: 'OLLAMA_API_KEY',
    scope: 'server',
  },
];

class OllamaProvider extends BaseModelProvider<OllamaConfig> {
  constructor(id: string, name: string, config: OllamaConfig) {
    super(id, name, config);
  }

  async getDefaultModels(): Promise<ModelList> {
    try {
      // Determine the base URL based on mode
      const baseURL =
        this.config.mode === 'cloud'
          ? 'https://ollama.com'
          : this.config.baseURL || 'http://localhost:11434';

      const headers: Record<string, string> = {
        'Content-type': 'application/json',
      };

      // Add Authorization header if in cloud mode and API key is provided
      if (this.config.mode === 'cloud' && this.config.apiKey) {
        headers['Authorization'] = `Bearer ${this.config.apiKey}`;
      }

      const res = await fetch(`${baseURL}/api/tags`, {
        method: 'GET',
        headers,
      });

      const data = await res.json();

      const models: Model[] = data.models.map((m: any) => {
        return {
          name: m.name,
          key: m.model,
        };
      });

      return {
        embedding: models,
        chat: models,
      };
    } catch (err) {
      if (err instanceof TypeError) {
        throw new Error(
          'Error connecting to Ollama API. Please ensure the base URL is correct and the Ollama server is running.',
        );
      }

      throw err;
    }
  }

  async getModelList(): Promise<ModelList> {
    const defaultModels = await this.getDefaultModels();
    const configProvider = getConfiguredModelProviderById(this.id)!;

    return {
      embedding: [
        ...defaultModels.embedding,
        ...configProvider.embeddingModels,
      ],
      chat: [...defaultModels.chat, ...configProvider.chatModels],
    };
  }

  async loadChatModel(key: string): Promise<BaseLLM<any>> {
    const modelList = await this.getModelList();

    const exists = modelList.chat.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading Ollama Chat Model. Invalid Model Selected',
      );
    }

    // Determine the base URL based on mode
    const baseURL =
      this.config.mode === 'cloud'
        ? 'https://ollama.com'
        : this.config.baseURL || 'http://localhost:11434';

    return new OllamaLLM({
      baseURL,
      model: key,
      apiKey: this.config.mode === 'cloud' ? this.config.apiKey : undefined,
    });
  }

  async loadEmbeddingModel(key: string): Promise<BaseEmbedding<any>> {
    const modelList = await this.getModelList();
    const exists = modelList.embedding.find((m) => m.key === key);

    if (!exists) {
      throw new Error(
        'Error Loading Ollama Embedding Model. Invalid Model Selected.',
      );
    }

    // Determine the base URL based on mode
    const baseURL =
      this.config.mode === 'cloud'
        ? 'https://ollama.com'
        : this.config.baseURL || 'http://localhost:11434';

    return new OllamaEmbedding({
      model: key,
      baseURL,
      apiKey: this.config.mode === 'cloud' ? this.config.apiKey : undefined,
    });
  }

  static parseAndValidate(raw: any): OllamaConfig {
    if (!raw || typeof raw !== 'object')
      throw new Error('Invalid config provided. Expected object');

    const mode = raw.mode || 'local';

    if (mode !== 'local' && mode !== 'cloud') {
      throw new Error('Invalid mode. Must be "local" or "cloud"');
    }

    // Validate based on mode
    if (mode === 'cloud' && !raw.apiKey) {
      throw new Error(
        'API Key is required for Ollama Cloud mode. Get one at https://ollama.com',
      );
    }

    return {
      mode: mode as 'local' | 'cloud',
      baseURL: raw.baseURL ? String(raw.baseURL) : undefined,
      apiKey: raw.apiKey ? String(raw.apiKey) : undefined,
    };
  }

  static getProviderConfigFields(): UIConfigField[] {
    return providerConfigFields;
  }

  static getProviderMetadata(): ProviderMetadata {
    return {
      key: 'ollama',
      name: 'Ollama',
    };
  }
}

export default OllamaProvider;
