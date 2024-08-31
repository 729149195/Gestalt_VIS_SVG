import { createStore } from 'vuex';

export default createStore({
  state() {
    return {
      currentPreviewFileName: null, // 当前预览的文件名
      gmInfoData: null,              // 一些 GM 信息数据
      loading: false,                // 加载状态
      selectedSvg: null,             // 当前选中的 SVG 内容
      selectedNodes: {               // 当前选中的节点信息
        nodeIds: [],                 // 选中的节点 ID 列表
        group: null                  // 选中的组（可选）
      },
      AllVisiableNodes: [],          // 所有可见的节点 ID 列表
      ele_num_data: null             // 元素数量数据
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
    UPDATE_SELECTED_NODES(state, payload) {
      // 更新选中的节点 ID 列表和组信息
      state.selectedNodes.nodeIds = payload.nodeIds;
      state.selectedNodes.group = payload.group || null;
      // console.log(state.selectedNodes.nodeIds)
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
    }
  }
});
