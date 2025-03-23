/* 自定义Element UI组件样式 */
:deep(.el-color-picker__trigger) {
  border-color: rgba(255, 255, 255, 0.2);
}

:deep(.el-input-number) {
  width: 80px !important;
  background: white !important;
  border-radius: 4px !important;
  border: 1px solid #dcdfe6 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
}

:deep(.el-input-number .el-input) {
  display: none !important;
}

:deep(.el-input-number .el-input-number__decrease),
:deep(.el-input-number .el-input-number__increase) {
  background: white !important;
  color: #606266 !important;
  border-color: #dcdfe6 !important;
  height: 100% !important;
  width: 40px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  top: 0 !important;
  margin: 0 !important;
}

:deep(.el-input-number .el-input-number__decrease:hover),
:deep(.el-input-number .el-input-number__increase:hover) {
  color: #409EFF !important;
}

:deep(.el-select) {
  width: 70px;
}

:deep(.el-select .el-input__inner) {
  color: #606266;
  background: white;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}

/* 确保下拉菜单在最顶层 */
:deep(.el-select__popper),
:deep(.el-color-dropdown),
:deep(.el-popper) {
  z-index: 1000001 !important;
  background-color: white !important;
  border-color: #e4e7ed !important;
  color: #606266 !important;
}

/* 添加编辑工具提示样式 */
.edit-tooltip {
  position: fixed !important;
  background-color: white !important;
  color: #606266 !important;
  padding: 12px !important;
  border-radius: 8px !important;
  z-index: 1000000 !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
  min-width: 200px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  gap: 12px !important;
  pointer-events: auto !important;
  opacity: 1 !important;
  margin: 0 !important;
  border: 1px solid #e4e7ed !important;
  backdrop-filter: blur(8px) !important;
  transition: all 0.2s ease !important;
}

/* 箭头样式 */
.edit-tooltip::after {
  content: '' !important;
  position: absolute !important;
  border: 8px solid transparent !important;
}

.edit-tooltip[data-position="top"]::after {
  bottom: -16px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  border-top-color: white !important;
}

.edit-tooltip[data-position="bottom"]::after {
  top: -16px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  border-bottom-color: white !important;
}

.edit-tooltip[data-position="left"]::after {
  right: -16px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  border-left-color: white !important;
}

.edit-tooltip[data-position="right"]::after {
  left: -16px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  border-right-color: white !important;
}

.edit-tooltip-header {
  font-size: 14px !important;
  font-weight: 500 !important;
  margin-bottom: 4px !important;
  color: #303133 !important;
  width: 100% !important;
  text-align: center !important;
  padding-bottom: 6px !important;
  border-bottom: 1px solid #e4e7ed !important;
}

.edit-tooltip-footer {
  font-size: 12px !important;
  color: #909399 !important;
  margin-top: 4px !important;
  text-align: center !important;
  width: 100% !important;
  padding-top: 6px !important;
  border-top: 1px solid #e4e7ed !important;
}

/* 数值编辑器容器样式 */
:deep(.number-editor) {
  background: white !important;
  border-radius: 4px !important;
  padding: 0 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

/* 确保颜色选择器在白色背景下正常显示 */
:deep(.color-editor) {
  background: white !important;
  border-radius: 4px !important;
  padding: 8px !important;
}

/* 修改工具提示内所有文本颜色 */
.edit-tooltip * {
  color: #606266 !important;
} 