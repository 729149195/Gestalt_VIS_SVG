<template>
  <div class="common-layout">
    <el-container class="full-height">
      <el-header class="header">
        <h2 class="title">图形模式感知问卷</h2>
      </el-header>
      <el-divider>
        <el-icon><star-filled /></el-icon>
      </el-divider>
      <el-main>
        <div class="personal-info">
          <h2>个人信息</h2>
          <p class="confidentiality">您的信息将被保密。( •̀ ω •́ )✧</p>
          <el-form label-position="top" class="form" @submit.prevent="handleSubmit">
            <el-form-item label="1、学号" label-for="studentid">
              <el-input id="studentid" v-model="form.studentid" placeholder="输入您的学号" class="input-field"></el-input>
            </el-form-item>
            <el-form-item label="2、年龄" label-for="age">
              <el-input id="age" v-model="form.age" placeholder="输入您的年龄" class="input-field"></el-input>
            </el-form-item>
            <el-form-item label="3、性别" label-for="gender">
              <el-radio-group id="gender" v-model="form.gender" class="input-field">
                <el-radio :value="'male'">男</el-radio>
                <el-radio :value="'female'">女</el-radio>
                <!-- <el-radio :value="OTHER">其他</el-radio> -->
              </el-radio-group>
              <!-- <span v-if="form.gender === OTHER">你确定吗？(ﾟДﾟ*)ﾉ</span> -->
            </el-form-item>
            <el-form-item label="4、您是否有视觉感知障碍（如色盲、色弱等）？" label-for="visualimpairment">
              <el-radio-group id="visualimpairment" v-model="form.visualimpairment" class="input-field">
                <el-radio :value="'yes'">有</el-radio>
                <el-radio :value="'no'">没有</el-radio>
              </el-radio-group>
              <span v-if="form.visualimpairment === 'yes'" style="color: red;">非常抱歉，您无法参与本次实验இ௰இ</span>
            </el-form-item>
            <el-form-item label="5、您是否有独立创建过可视化作品并将其实际应用？（例如 财务报表 等）" label-for="visualizationExperience">
              <el-radio-group id="visualizationExperience" v-model="form.visualizationExperience" class="input-field">
                <el-radio :value="'yes'">有</el-radio>
                <el-radio :value="'no'">没有</el-radio>
              </el-radio-group>
            </el-form-item>
            <el-form-item class="form-button">
              <div>
                <el-button @click="handleClean">清空</el-button>
                <el-button type="primary" @click="handleSubmit">下一步</el-button>
              </div>
            </el-form-item>
          </el-form>
        </div>
      </el-main>
      <el-divider border-style="double" />
      <el-footer>{{ currentTime }}</el-footer>
    </el-container>
  </div>
  <el-dialog v-model="DialogVisible" title="问卷介绍（预估完成所需时间为20分钟）" width="750">
    <img style="width: 100%; margin-top: 10px" src="/img/introduction.png" alt="Wechat QR Code">
    <template #footer>
      <div class="dialog-footer">
        <!-- <el-button @click="showdata"> 实例总览 </el-button> -->
        <el-button @click="DialogVisible = false" type="primary">开始</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { useStore } from 'vuex';
import { useRouter } from 'vue-router';
import { StarFilled } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { getSubmissionCount } from '../api/counter';

const store = useStore();
const router = useRouter();
const DialogVisible = ref(true)

const currentTime = ref(new Date().toLocaleTimeString());

const showdata = () => {
  router.push('/showdata');
};


const updateCurrentTime = () => {
  currentTime.value = new Date().toLocaleTimeString();
};

let timer = null;

onMounted(async () => {
  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
    return;
  }
  timer = setInterval(updateCurrentTime, 1000);
});

onUnmounted(() => {
  clearInterval(timer);
});

const form = ref({
  studentid: '',
  age: '',
  gender: '',
  visualimpairment: '',
  visualizationExperience: '',
});

const OTHER = 'other';  // Define a constant for 'other'

const handleSubmit = async () => {
  const age = parseInt(form.value.age);
  if (!form.value.studentid) {
    ElMessage({
      message: '请填写学号。',
      type: 'warning',
    });
    return;
  }

  if (!age || isNaN(age) || age < 10 || age > 90) {
    ElMessage({
      message: '请输入有效的年龄（14到85岁之间的整数）',
      type: 'warning',
    });
    return;
  }

  if (!form.value.gender) {
    ElMessage({
      message: '请选择性别。',
      type: 'warning',
    });
    return;
  }

  if (!form.value.visualimpairment) {
    ElMessage({
      message: '请选择是否有视觉感知障碍。',
      type: 'warning',
    });
    return;
  }

  if (form.value.visualimpairment === 'yes') {
    ElMessage({
      message: '抱歉您无法参与本次实验〒▽〒',
      type: 'warning',
    });
    return;
  }

  if (!form.value.visualizationExperience) {
    ElMessage({
      message: '请选择是否有可视化经验。',
      type: 'warning',
    });
    return;
  }

  const count = await getSubmissionCount();
  if (count >= 50) {
    router.push('/limit-reached');
    return;
  }

  store.dispatch('submitForm', form.value);
  router.push('/questionstest');
};

const handleClean = () => {
  form.value = {
    studentid: '',
    age: '',
    gender: '',
    visualimpairment: '',
    visualizationExperience: '',
  };
};

</script>

<style lang="scss" scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#tsparticles {
  position: absolute;
  width: 100%;
  height: 100%;
  z-index: -1;
}

.common-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 80vw;
  margin: 0 auto;
  position: relative; // Ensure the layout is positioned correctly
}

.full-height {
  display: flex;
  flex-direction: column;
  flex: 1;
}

.el-header,
.el-footer {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.el-main {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: start;
  position: relative;
  top: 10px;
  flex-direction: column;
  padding: 10px;
}

.personal-info {
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
  padding: 30px;
  background: #f9f9f9;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.personal-info h2 {
  font-size: 1.7rem;
  text-align: center;
  color: #333;
}

.confidentiality {
  font-size: 0.8rem;
  color: #666;
  text-align: center;
  margin-bottom: 20px;
}

.form-button {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  .el-button {
    width: 14rem;
    height: 40px;
    font-size: 16px;
  }
}

.el-radio {
  margin-right: 30px;
}

.el-form-item {
  margin-bottom: 20px;
}

.el-select-dropdown__item {
  padding-left: 10px;
}

.title {
  font-size: 2.5rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 2px;
  text-align: center;
  background: linear-gradient(90deg, rgb(21, 250, 250) 0%, rgb(25, 57, 242) 50%, rgba(58, 123, 213, 1) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 5px;
}

.dialog-footer {
  .el-button {
    width: 60px;
  }
}
</style>
