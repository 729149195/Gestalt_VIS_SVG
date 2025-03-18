import { createStore } from 'vuex';

const generateRandomId = () => {
  return Math.floor(100000 + Math.random() * 900000).toString();
};

// 生成不重复的随机数数组
// const generateRandomArray = () => {
//   const numbers = Array.from({ length: 20 }, (_, index) => index + 1);
//   const randomArray = [];
//   while (randomArray.length < 10) {
//     const randomIndex = Math.floor(Math.random() * numbers.length);
//     const number = numbers.splice(randomIndex, 1)[0];
//     randomArray.push(number);
//   }
//   return randomArray;
// };
const generateRandomArray = () => {
  return Array.from({ length: 20 }, (_, index) => index + 1);
};

const store = createStore({
  state() {
    return {
      formData: {},
      selectedNodes: {
        nodeIds: [],
        group: null
      },
      AllVisiableNodes: [],
      groups: {},
      ratings: {},
      steps: [], // 用于存储随机生成的不重复数字
      startTime: null, // 记录开始时间以计算持续时间
      totalTimeSpent: 0, // 总共花费的时间（以秒为单位）
      submittedData: null, // 新增：存储提交的数据
    };
  },
  mutations: {
    setFormData(state, data) {
      state.formData = { ...data, id: generateRandomId() };
      state.startTime = new Date().toISOString();
    },
    setSteps(state, steps) {
      state.steps = steps;
    },
    UPDATE_SELECTED_NODES(state, payload) {
      state.selectedNodes.nodeIds = payload.nodeIds;
      state.selectedNodes.group = payload.group;
    },
    UPDATE_ALL_VISIABLE_NODES(state, nodeIds) {
      state.AllVisiableNodes = nodeIds;
    },
    ADD_NODE_TO_GROUP(state, payload) {
      const { step, group, nodeId } = payload;
      if (!state.groups[step]) {
        state.groups[step] = {};
      }
      if (!state.groups[step][group]) {
        state.groups[step][group] = [];
      }
      if (!state.groups[step][group].includes(nodeId)) {
        state.groups[step][group].push(nodeId);
      }
    },
    REMOVE_NODE_FROM_GROUP(state, payload) {
      const { step, group, nodeId } = payload;
      const index = state.groups[step]?.[group]?.indexOf(nodeId);
      if (index !== -1) {
        state.groups[step][group].splice(index, 1);
      }
    },
    ADD_NEW_GROUP(state, payload) {
      const { step, group } = payload;
      if (!state.groups[step]) {
        state.groups[step] = {};
      }
      if (!state.groups[step][group]) {
        state.groups[step][group] = [];
      }
    },
    UPDATE_RATING(state, payload) {
      const { step, group, rating, type } = payload;
      if (!state.ratings[step]) {
        state.ratings[step] = {};
      }
      if (!state.ratings[step][group]) {
        state.ratings[step][group] = { attention: 1, correlation_strength: 1, exclusionary_force: 1 };
      }
      state.ratings[step][group][type] = rating;
    },
    ADD_OTHER_GROUP(state, payload) {
      const { step, group, nodeIds } = payload;
      if (!state.groups[step]) {
        state.groups[step] = {};
      }
      state.groups[step][group] = nodeIds;
    },
    DELETE_GROUP(state, payload) {
      const { step, group } = payload;
      if (state.groups[step] && state.groups[step][group]) {
        delete state.groups[step][group];
        // 重新分配组和评分
        const newGroups = {};
        const newRatings = {};
        Object.keys(state.groups[step]).forEach((grp, idx) => {
          const newGroup = `组合${idx + 1}`;
          newGroups[newGroup] = state.groups[step][grp];
          newRatings[newGroup] = state.ratings[step][grp];
        });
        state.groups[step] = newGroups;
        state.ratings[step] = newRatings;
      }
    },
    RESET_TRAINING_DATA(state) {
      state.selectedNodes.nodeIds = [];
      state.AllVisiableNodes = [];
      state.groups = {};
      state.ratings = {};
    },
    UPDATE_TOTAL_TIME_SPENT(state, time) {
      state.totalTimeSpent = time;
    },
    RESET_STEP_RATINGS(state, step) {
      state.ratings[step] = {};
    },
    CLEAR_FORM_DATA(state) {
      state.formData = null;
    },
    SET_SUBMITTED_DATA(state, data) {
      state.submittedData = data;
    },
  },
  actions: {
    submitForm({ commit }, data) {
      commit('setFormData', data);
    },
    initializeSteps({ commit }) {
      const randomArray = generateRandomArray();
      commit('setSteps', randomArray);
    },
  },
  getters: {
    getFormData: (state) => state.formData,
    getGroups: (state) => (step) => state.groups[step] || {},
    getRating: (state) => (step, group, type) => state.ratings[step]?.[group]?.[type] || 1,
    getTotalTimeSpent: (state) => state.totalTimeSpent
  },
});

export default store;
