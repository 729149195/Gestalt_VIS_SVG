.element-type-table td {
    padding: 0 4px;
    text-align: center;
    vertical-align: middle;
    white-space: nowrap;
    font-size: 0.85em;
    line-height: 1;
    height: 33.33%;
}

.element-type-table td:first-child {
    text-align: left;
    font-weight: 600;
    color: #905F29 !important; /* 使用主题色 */
    width: 100px; /* 从170px减小到130px */
    padding-left: 0;
}

/* 添加第二列之后的单元格样式，减小间距 */
.element-type-table td:not(:first-child) {
    padding: 0 2px;
} 