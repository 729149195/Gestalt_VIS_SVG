import { createStore } from 'vuex';

export default createStore({
  state() {
    return {
      currentPreviewFileName: null, // 当前预览的文件名
      gmInfoData: null,              // 一些 GM 信息数据
      loading: false,                // 加载状态
      selectedSvg: null,             // 当前选中的 SVG 内容
      displaySvgContent: null,       // S区显示的SVG内容
      selectedNodes: {               // 当前选中的节点信息
        nodeIds: [],                 // 选中的节点 ID 列表
        group: null                  // 选中的组（可选）
      },
      AllVisiableNodes: [],          // 所有可见的节点 ID 列表
      ele_num_data: null,            // 元素数量数据
      visualSalience: [],            // 视觉显著性数组，存储所有卡片的显著性值
      copiedValue: null,             // 存储复制的值
      copiedValueType: null,         // 存储复制值的类型（color, stroke-width, area）
      copiedFeatureName: null        // 存储复制值的特征名称
    };
  },
  mutations: {
    setCurrentPreviewFileName(state, filename) {
      state.currentPreviewFileName = filename;
    },
    setGMInfoData(state, data) {
      state.gmInfoData = data;
    },
    setLoading(state, isLoading) {
      state.loading = isLoading;
    },
    setSelectedSvg(state, svgContent) {
      state.selectedSvg = svgContent;
    },
    SET_DISPLAY_SVG_CONTENT(state, svgContent) {
      state.displaySvgContent = svgContent;
    },
    UPDATE_SELECTED_NODES(state, payload) {
      // 更新选中的节点 ID 列表和组信息
      state.selectedNodes.nodeIds = payload.nodeIds;
      state.selectedNodes.group = payload.group || null;
    },
    SET_ALL_VISIBLE_NODES(state, nodeIds) {
      // 设置所有可见的节点 ID 列表
      state.AllVisiableNodes = nodeIds;
    },
    GET_ELE_NUM_DATA(state, data) {
      state.ele_num_data = data;
    },
    CLEAR_SELECTED_NODES(state) {
      // 清空选中的节点和组信息
      state.selectedNodes.nodeIds = [];
      state.selectedNodes.group = null;
    },
    ADD_SELECTED_NODE(state, newNodeId) {
      // 替换整个数组，确保触发 Vue 的响应式系统
      state.selectedNodes.nodeIds = [...state.selectedNodes.nodeIds, newNodeId];
    },
    REMOVE_SELECTED_NODE(state, nodeIdToRemove) {
      state.selectedNodes.nodeIds = state.selectedNodes.nodeIds.filter(id => id !== nodeIdToRemove);
    },
    SET_VISUAL_SALIENCE(state, value) {
      // 设置视觉显著性数组
      state.visualSalience = value;
    },
    SET_COPIED_VALUE(state, { value, type, featureName }) {
      state.copiedValue = value;
      state.copiedValueType = type;
      state.copiedFeatureName = featureName;
    },
    CLEAR_COPIED_VALUE(state) {
      state.copiedValue = null;
      state.copiedValueType = null;
      state.copiedFeatureName = null;
    }
  },
  actions: {
    updateSelectedNodes({ commit }, payload) {
      commit('UPDATE_SELECTED_NODES', payload);
    },
    clearSelectedNodes({ commit }) {
      commit('CLEAR_SELECTED_NODES');
    },
    addSelectedNode({ commit }, nodeId) {
      commit('ADD_SELECTED_NODE', nodeId);
    },
    removeSelectedNode({ commit }, nodeId) {
      commit('REMOVE_SELECTED_NODE', nodeId);
    },
    setCopiedValue({ commit }, payload) {
      commit('SET_COPIED_VALUE', payload);
    },
    clearCopiedValue({ commit }) {
      commit('CLEAR_COPIED_VALUE');
    }
  },
  getters: {
    getSelectedNodes(state) {
      // 获取当前选中的节点 ID 列表
      return state.selectedNodes.nodeIds;
    },
    isNodeSelected: (state) => (nodeId) => {
      // 判断某个节点是否被选中
      return state.selectedNodes.nodeIds.includes(nodeId);
    },
    getSelectedGroup(state) {
      // 获取当前选中的组信息
      return state.selectedNodes.group;
    },
    getCopiedValue(state) {
      return {
        value: state.copiedValue,
        type: state.copiedValueType,
        featureName: state.copiedFeatureName
      };
    }
  }
});
