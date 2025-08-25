/**
 * 新闻分类系统主应用组件
 * 提供新闻文本分类的用户界面
 */
import React, { useState, useEffect } from 'react';
import {
  Layout,
  Typography,
  Card,
  Input,
  Button,
  Result,
  Progress,
  Tag,
  Space,
  Divider,
  List,
  message,
  Statistic,
  Row,
  Col,
} from 'antd';
import {
  SendOutlined,
  ClearOutlined,
  InfoCircleOutlined,
  FileTextOutlined,
} from '@ant-design/icons';
import { newsApi } from './services/api';
import type { NewsCategory, NewsItem, ModelInfoResponse } from './types';
import './App.css';

const { Header, Content, Footer } = Layout;
const { Title, Paragraph, Text } = Typography;
const { TextArea } = Input;

// 类别颜色映射
const categoryColors: Record<NewsCategory, string> = {
  '教育': 'blue',
  '科技': 'green',
  '社会': 'orange',
  '时政': 'red',
  '财经': 'purple',
  '房产': 'cyan',
  '家居': 'magenta',
};

function App() {
  const [inputText, setInputText] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionHistory, setPredictionHistory] = useState<NewsItem[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [categories, setCategories] = useState<NewsCategory[]>([]);

  /**
   * 组件挂载时获取模型信息和类别列表
   */
  useEffect(() => {
    fetchModelInfo();
    fetchCategories();
  }, []);

  /**
   * 获取模型信息
   */
  const fetchModelInfo = async () => {
    try {
      const info = await newsApi.getModelInfo();
      setModelInfo(info);
    } catch (error) {
      message.error('获取模型信息失败');
      console.error('获取模型信息错误:', error);
    }
  };

  /**
   * 获取类别列表
   */
  const fetchCategories = async () => {
    try {
      const response = await newsApi.getCategories();
      setCategories(response.categories);
    } catch (error) {
      message.error('获取类别列表失败');
      console.error('获取类别列表错误:', error);
    }
  };

  /**
   * 执行新闻分类预测
   */
  const handlePredict = async () => {
    if (!inputText.trim()) {
      message.warning('请输入新闻文本');
      return;
    }

    setIsLoading(true);
    const newItem: NewsItem = {
      id: Date.now().toString(),
      text: inputText,
      isProcessing: true,
    };

    // 添加到历史记录并显示处理中状态
    setPredictionHistory(prev => [newItem, ...prev]);

    try {
      const result = await newsApi.predictSingle(inputText);
      
      // 更新预测结果
      setPredictionHistory(prev => 
        prev.map(item => 
          item.id === newItem.id 
            ? {
                ...item,
                predictedCategory: result.predicted_category,
                confidence: result.confidence,
                isProcessing: false,
              }
            : item
        )
      );

      message.success(`预测完成: ${result.predicted_category}`);
    } catch (error) {
      // 更新错误状态
      setPredictionHistory(prev => 
        prev.map(item => 
          item.id === newItem.id 
            ? { ...item, isProcessing: false }
            : item
        )
      );
      
      message.error('预测失败，请重试');
      console.error('预测错误:', error);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * 清空输入和历史记录
   */
  const handleClear = () => {
    setInputText('');
    setPredictionHistory([]);
  };

  /**
   * 渲染预测历史记录项
   */
  const renderHistoryItem = (item: NewsItem) => {
    if (item.isProcessing) {
      return (
        <List.Item>
          <div style={{ width: '100%' }}>
            <div style={{ marginBottom: 8 }}>
              <FileTextOutlined /> 正在分析: {item.text.substring(0, 50)}...
            </div>
            <Progress percent={undefined} status="active" />
          </div>
        </List.Item>
      );
    }

    return (
      <List.Item>
        <div style={{ width: '100%' }}>
          <div style={{ marginBottom: 8 }}>
            <Text strong>原文：</Text>
            <Text>{item.text}</Text>
          </div>
          {item.predictedCategory && (
            <Space size="middle">
              <Tag color={categoryColors[item.predictedCategory]} style={{ fontSize: '14px', padding: '4px 12px' }}>
                {item.predictedCategory}
              </Tag>
              <Text type="secondary">
                置信度: {(item.confidence! * 100).toFixed(1)}%
              </Text>
            </Space>
          )}
        </div>
      </List.Item>
    );
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <Title level={2} style={{ margin: 0, color: '#1890ff' }}>
          📰 新闻分类系统
        </Title>
      </Header>

      <Content style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
        {/* 系统信息 */}
        {modelInfo && (
          <Card style={{ marginBottom: 24 }}>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="模型状态"
                  value={modelInfo.status}
                  prefix={<InfoCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="模型类型"
                  value={modelInfo.model_type}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="支持类别"
                  value={modelInfo.categories.length}
                  suffix="个"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="特征维度"
                  value={modelInfo.feature_count || 0}
                />
              </Col>
            </Row>
          </Card>
        )}

        {/* 类别展示 */}
        <Card title="支持的新闻类别" style={{ marginBottom: 24 }}>
          <Space wrap>
            {categories.map(category => (
              <Tag key={category} color={categoryColors[category]} style={{ fontSize: '14px', padding: '4px 12px' }}>
                {category}
              </Tag>
            ))}
          </Space>
        </Card>

        {/* 输入区域 */}
        <Card title="新闻文本分类" style={{ marginBottom: 24 }}>
          <Space.Compact style={{ width: '100%', marginBottom: 16 }}>
            <TextArea
              placeholder="请输入新闻标题或内容，例如：教育部发布新的课程标准..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              rows={4}
              maxLength={1000}
              showCount
              style={{ flex: 1 }}
              onPressEnter={(e) => {
                if (e.ctrlKey || e.metaKey) {
                  handlePredict();
                }
              }}
            />
          </Space.Compact>
          
          <Space>
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={handlePredict}
              loading={isLoading}
              disabled={!inputText.trim()}
            >
              开始分类
            </Button>
            <Button
              icon={<ClearOutlined />}
              onClick={handleClear}
              disabled={isLoading}
            >
              清空
            </Button>
          </Space>
          
          <Paragraph type="secondary" style={{ marginTop: 8, marginBottom: 0 }}>
            提示：支持 Ctrl+Enter 快速提交
          </Paragraph>
        </Card>

        {/* 预测历史 */}
        {predictionHistory.length > 0 && (
          <Card title={`预测历史 (${predictionHistory.length})`}>
            <List
              dataSource={predictionHistory}
              renderItem={renderHistoryItem}
              size="large"
            />
          </Card>
        )}

        {/* 使用说明 */}
        {predictionHistory.length === 0 && (
          <Card>
            <Result
              icon={<FileTextOutlined />}
              title="欢迎使用新闻分类系统"
              subTitle="输入新闻文本，系统将自动识别其所属类别"
              extra={
                <div>
                  <Paragraph>
                    本系统支持以下7个类别的中文新闻分类：
                  </Paragraph>
                  <Space wrap style={{ justifyContent: 'center' }}>
                    {categories.map(category => (
                      <Tag key={category} color={categoryColors[category]} style={{ fontSize: '14px', padding: '4px 12px' }}>
                        {category}
                      </Tag>
                    ))}
                  </Space>
                </div>
              }
            />
          </Card>
        )}
      </Content>

      <Footer style={{ textAlign: 'center', color: '#666' }}>
        新闻分类系统 ©2024 基于机器学习技术
      </Footer>
    </Layout>
  );
}

export default App;
