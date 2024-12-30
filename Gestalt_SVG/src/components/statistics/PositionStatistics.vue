<template>
    <span style="color: #666">{{ title }}</span>
    <div ref="chartContainer" style="width: 100%; height: calc(70%);"></div>
</template>

<script setup>
import { onMounted, ref, defineProps } from 'vue';
import * as d3 from 'd3';
import { useStore } from 'vuex';

const props = defineProps({
    position: {
        type: String,
        required: true,
        validator: (value) => ['top', 'bottom', 'left', 'right'].includes(value)
    },
    title: {
        type: String,
        default: 'Position Statistics'
    }
});

const store = useStore();
const chartContainer = ref(null);


const eleURL = `http://localhost:5000/${props.position}_position`;

onMounted(async () => {
    if (!chartContainer.value) return;

    try {
        const response = await fetch(eleURL);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        const dataset = Object.keys(data).map((range) => ({
            range,
            tags: data[range].tags,
            totals: Object.entries(data[range].total).map(([key, value]) => ({ key, value }))
        }));
        renderChart(dataset);
    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
});

const renderChart = (dataset) => {
    const container = chartContainer.value;
    const svgWidth = container.clientWidth;
    const svgHeight = container.clientHeight;
    const margin = { 
        top: svgHeight * 0.08, 
        right: svgWidth * 0.02, 
        bottom: svgHeight * 0.35, 
        left: svgWidth * 0.12 
    };
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    const xScale = d3.scaleBand()
        .range([0, width])
        .padding(0.2)
        .domain(dataset.map(d => d.range));

    const yScale = d3.scaleLinear()
        .range([height, 0])
        .domain([0, d3.max(dataset, d => d3.sum(d.totals, t => t.value))]).nice();

    const svg = d3.select(chartContainer.value).append('svg')
        .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
        .attr('width', svgWidth)
        .attr('height', svgHeight)
        .attr('style', 'max-width: 100%; height: auto;')
        .style("position", "relative");

    const tooltip = d3.select(chartContainer.value)
        .append("div")
        .attr("class", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden")
        .style("background", "white")
        .style("border", "1px solid #ddd")
        .style("padding", "10px")
        .style("border-radius", "2px")
        .style("pointer-events", "none")
        .style("box-shadow", "0px 0px 10px rgba(0,0,0,0.1)")
        .style("white-space", "nowrap");

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .selectAll("text")
        .style("text-anchor", "end")
        .style("pointer-events", "none")
        .style("font-size", "12px")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-45)");

    const yAxis = g.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale))
        .call(g => g.selectAll('.tick line')
            .clone()
            .attr('x2', width)
            .attr('stroke', '#ddd'));

    const zoom = d3.zoom()
        .scaleExtent([1, 3])
        .translateExtent([[0, 0], [width, height]])
        .extent([[0, 0], [width, height]])
        .on('zoom', (event) => {
            const transform = event.transform;
            xScale.range([0, width].map(d => transform.applyX(d)));
            g.select('.x-axis').call(d3.axisBottom(xScale));
            g.selectAll('.range').attr('transform', d => `translate(${xScale(d.range)},0)`);
            g.selectAll('.range rect').attr('width', xScale.bandwidth());
        });

    svg.call(zoom);

    const rangeGroup = g.selectAll('.range')
        .data(dataset)
        .enter().append('g')
        .attr('class', 'range')
        .attr('transform', d => `translate(${xScale(d.range)},0)`)
        .attr("style", "cursor: pointer;")
        .on('mouseover', (event, d) => {
            tooltip.style("visibility", "visible")
                .html(() => {
                    const tagsContent = d.tags.map(tag => `${tag}<br/>`).join("");
                    const content = `<strong>Range:</strong> ${d.range}<br/><strong>Tags:</strong><br/>${tagsContent}`;
                    return content;
                });
            const tooltipHeight = tooltip.node().getBoundingClientRect().height;
            tooltip.style("top", (event.pageY - tooltipHeight - 10) + "px")
                .style("left", (event.pageX + 10) + "px");
        })
        .on('mouseout', () => {
            tooltip.style("visibility", "hidden");
        })
        .on('click', (event, d) => {
            store.commit('UPDATE_SELECTED_NODES', { nodeIds: d.tags, group: null });
        });

    const customColorMap = {
        "circle": "#FFE119", "rect": "#E6194B", "line": "#4363D8",
        "polyline": "#911EB4", "polygon": "#F58231", "path": "#3CB44B",
        "text": "#46F0F0", "ellipse": "#F032E6", "image": "#BCF60C",
        "use": "#FFD700", "defs": "#FF4500", "linearGradient": "#1E90FF",
        "radialGradient": "#FF6347", "stop": "#4682B4", "symbol": "#D2691E",
        "clipPath": "#FABEBE", "mask": "#8B008B", "pattern": "#A52A2A",
        "filter": "#5F9EA0", "feGaussianBlur": "#D8BFD8", "feOffset": "#FFDAB9",
        "feBlend": "#32CD32", "feFlood": "#FFD700", "feImage": "#FF6347",
        "feComposite": "#FF4500", "feColorMatrix": "#1E90FF", "feMerge": "#FF1493",
        "feMorphology": "#00FA9A", "feTurbulence": "#8B008B",
        "feDisplacementMap": "#FFD700", "unknown": "#696969"
    };

    rangeGroup.each(function (d) {
        let yAccumulator = 0;
        d3.select(this).selectAll('rect')
            .data(d.totals)
            .enter().append('rect')
            .attr('x', 0)
            .attr('y', t => {
                const y = yScale(yAccumulator + t.value);
                yAccumulator += t.value;
                return y;
            })
            .attr('width', xScale.bandwidth())
            .attr('height', t => height - yScale(t.value))
            .attr('fill', t => customColorMap[t.key] || '#696969')
            .attr('rx', 2)
            .attr('ry', 2);
    });

    svg.append("text")
        .attr("transform", `translate(${width / 2},${height - 5})`)
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .attr("dx", "10em")
        .attr("dy", "8em")
        .text("Attributes")
        .text("Position zones");

        svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 15)
        .attr("x", 0 - (height))
        .style("text-anchor", "middle")
        .style("font-size", "14px")
        .attr("dx", "4.0em")
        .attr("dy", "0em")
        .text("Position Number");
};
</script>

<style scoped>
.tooltip strong {
    color: blue;
}

.tooltip span {
    color: black;
}
</style> 