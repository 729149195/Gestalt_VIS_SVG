const API_BASE_URL = '/api';

// 获取当前计数
export const getSubmissionCount = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/count`);
    const data = await response.json();
    return data.count;
  } catch (error) {
    console.error('Error fetching count:', error);
    return 0;
  }
};

// 增加计数
export const incrementCount = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/increment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    const data = await response.json();
    return data.count;
  } catch (error) {
    console.error('Error incrementing count:', error);
    return 0;
  }
};

// 重置计数
export const resetSubmissionCount = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/reset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    const data = await response.json();
    return data.count;
  } catch (error) {
    console.error('Error resetting count:', error);
    return 0;
  }
}; 