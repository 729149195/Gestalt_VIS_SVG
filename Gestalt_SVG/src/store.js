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
      clickedCardSalience: null,     // 存储用户点击的特定卡片的显著性值
      copiedValue: null,             // 存储复制的值
      copiedValueType: null,         // 存储复制值的类型（color, stroke-width, area）
      copiedFeatureName: null,        // 存储复制值的特征名称
      revelioGoodClusters: [],        // 存储reveliogood_n类的ID组，二维数组
      revelioBadClusters: []         // 存储reveliobad_n类的ID组，二维数组
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
      state.selectedNodes = payload;
    },
    SET_ALL_VISIBLE_NODES(state, nodeIds) {
      state.AllVisiableNodes = nodeIds;
    },
    GET_ELE_NUM_DATA(state, data) {
      state.ele_num_data = data;
    },
    CLEAR_SELECTED_NODES(state) {
      state.selectedNodes = {
        nodeIds: [],
        group: null
      };
    },
    ADD_SELECTED_NODE(state, nodeId) {
      if (!state.selectedNodes.nodeIds.includes(nodeId)) {
        state.selectedNodes.nodeIds.push(nodeId);
      }
    },
    REMOVE_SELECTED_NODE(state, nodeId) {
      state.selectedNodes.nodeIds = state.selectedNodes.nodeIds.filter(id => id !== nodeId);
    },
    SET_VISUAL_SALIENCE(state, value) {
      state.visualSalience = value;
    },
    SET_CLICKED_CARD_SALIENCE(state, value) {
      state.clickedCardSalience = value;
    },
    SET_COPIED_VALUE(state, payload) {
      state.copiedValue = payload.value;
      state.copiedValueType = payload.type;
      state.copiedFeatureName = payload.featureName;
    },
    CLEAR_COPIED_VALUE(state) {
      state.copiedValue = null;
      state.copiedValueType = null;
      state.copiedFeatureName = null;
    },
    SET_REVELIOGOOD_CLUSTERS(state, clusters) {
      state.revelioGoodClusters = clusters;
    },
    SET_REVELIOBAD_CLUSTERS(state, clusters) {
      state.revelioBadClusters = clusters;
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
    },
    setRevelioGoodClusters({ commit }, clusters) {
      commit('SET_REVELIOGOOD_CLUSTERS', clusters);
    },
    setRevelioBadClusters({ commit }, clusters) {
      commit('SET_REVELIOBAD_CLUSTERS', clusters);
    },
    setClickedCardSalience({ commit }, value) {
      commit('SET_CLICKED_CARD_SALIENCE', value);
    }
  },
  getters: {
    getSelectedNodes(state) {
      return state.selectedNodes.nodeIds;
    },
    isNodeSelected: (state) => (nodeId) => {
      return state.selectedNodes.nodeIds.includes(nodeId);
    },
    getSelectedGroup(state) {
      return state.selectedNodes.group;
    },
    getCopiedValue(state) {
      return {
        value: state.copiedValue,
        type: state.copiedValueType,
        featureName: state.copiedFeatureName
      };
    },
    getRevelioGoodClusters(state) {
      return state.revelioGoodClusters;
    },
    getRevelioBadClusters(state) {
      return state.revelioBadClusters;
    },
    getClickedCardSalience(state) {
      return state.clickedCardSalience;
    },
    getCurrentVisualSalience(state) {
      // 优先返回点击卡片的显著性值
      if (state.clickedCardSalience !== null) {
        return state.clickedCardSalience;
      }
      // 如果没有点击特定卡片，才使用通用逻辑
      if (Array.isArray(state.visualSalience) && state.visualSalience.length > 0) {
        return state.visualSalience[0].salienceValue;
      } 
      else if (typeof state.visualSalience === 'number') {
        return state.visualSalience;
      }
      return null;
    }
  }
});
