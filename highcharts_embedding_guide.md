# Highcharts Embedding Guide for Report Divs

This guide enables AI agents to create Highcharts visualizations as embedded `<div>` elements within reports, following established style conventions.

## Chart Types Covered

This guide provides explicit configurations for:

| Chart Type | Module Requirements | Data Format | Use Case |
|------------|---------------------|-------------|----------|
| **Column/Bar** | `highcharts.js` | `[y]` or `[[x, y]]` | Categorical comparisons, time series bars |
| **Stacked Bar** | `highcharts.js` | Multiple series arrays | Composition over categories |
| **Area** | `highcharts.js` | `[[x, y]]` | Time series, cumulative values |
| **Stacked Area** | `highcharts.js` | Multiple `[[x, y]]` series | Composition over time |
| **Line** | `highcharts.js` | `[[x, y]]` | Time series trends |
| **Combo (Bar+Line)** | `highcharts.js` | Mixed series types | Dual metrics with different scales |
| **Scatter** | `highcharts.js` | `[[x, y]]` | Correlation, distribution |
| **Bubble** | `highcharts.js`, `highcharts-more.js` | `[[x, y, z]]` | 3D relationships (x, y, size) |
| **Heatmap** | `heatmap.js`, `data.js`, `boost.js` | `[[x, y, value]]` | Temporal patterns, 2D grids |
| **Donut/Pie** | `highcharts.js` | `[{name, y}]` | Proportions, percentages |
| **Sankey** | `sankey.js` | Nodes + links `[from, to, weight]` | Flow diagrams, relationships |

## Core Structure for Embedded Divs

When embedding charts in reports (not standalone HTML files), use this minimal structure:

```html
<div id="chart-container-{unique-id}">
    <div id="chart-{unique-id}"></div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    // Embedded data as JavaScript constant
    const rawData = [
        {"DATE":"2025-10-01T00:00:00.000Z","VALUE":12345.67},
        // ... more rows
    ];
    
    // Data transformation (if needed)
    const chartData = rawData.map(item => ({
        x: new Date(item.DATE).getTime(),
        y: item.VALUE
    }));
    
    // Highcharts configuration
    Highcharts.chart('chart-{unique-id}', {
        // ... config (see sections below)
    });
});
</script>
```

**Key Differences from Standalone HTML:**
- No `<html>`, `<head>`, or `<body>` tags
- No Highcharts library imports (assumed loaded in parent document)
- No external CSS (rely on parent document styling)
- Use unique IDs per chart to avoid conflicts (e.g., `chart-container-volume`, `chart-volume`)
- Data embedded directly as JavaScript constants
- Wrap chart initialization in `DOMContentLoaded` event listener

**Critical:** Always use unique IDs. If creating multiple charts in one report, use descriptive suffixes:
- `chart-container-transfer-volume` / `chart-transfer-volume`
- `chart-container-gas-usage` / `chart-gas-usage`

## Color Palette

Use these predefined colors from `highcharts_rule.txt` when contextually appropriate:

### Primary Chain/Asset Colors
```javascript
const colors = {
    bitcoin: '#f7941d',
    ethereum: '#808080',
    avalanche: '#FFA07A',
    bsc: '#FFD700',
    near: '#00FF7F',
    solana: '#9945FF',
    polygon: '#8B5CF6',
    arbitrum: '#ADD8E6',
    optimism: '#8B0000',
    base: '#00008B',
    DAI: '#f4b82d',
    USDT: '#009393',
    USDC: '#2775ca'
};
```

**Usage:** Only use these colors when the context matches (e.g., USDC color for USDC-related data). For general contrasting colors, use the NEAR marketplace palette below or generate appropriate contrasting colors.

### High Contrast Palette (General Use)
```javascript
const highContrastColors = [
    '#FF2424', '#0000B9', '#F7A000', '#8486FF', 
    '#E087FF', '#86EFAC', '#F5DD6C'
];
```

## Standard Configuration Patterns

### Title and Subtitle
```javascript
title: {
    text: 'Chart Title',
    style: {
        fontSize: '1.5rem',
        fontWeight: 'bold',
        color: '#333333'
    }
},
subtitle: {
    text: 'Supporting context or data source attribution',
    style: {
        fontSize: '1rem',
        color: '#666666'
    }
}
```

### X-Axis (Date/Time)
```javascript
xAxis: {
    type: 'datetime',  // For time series
    title: {
        text: 'Date',
        style: { fontSize: '1.125rem' }
    },
    labels: {
        style: { fontSize: '1rem' },
        format: '{value:%b %e}',  // "Oct 1" format
        rotation: 0  // or -45 for crowded labels
    },
    tickInterval: 432000000  // Optional: 5 days in milliseconds
}
```

### X-Axis (Categories)
```javascript
xAxis: {
    categories: ['Category1', 'Category2', ...],
    title: {
        text: 'Category Label',
        style: { fontSize: '1.125rem' }
    },
    labels: {
        style: { fontSize: '1rem' },
        rotation: 0
    }
}
```

### Y-Axis (Numeric)
```javascript
yAxis: {
    title: {
        text: 'Value Label',
        style: {
            fontSize: '1.125rem',
            fontWeight: 'bold'
        }
    },
    labels: {
        style: { fontSize: '1rem' },
        format: '{value:,.0f}'  // Comma separators, no decimals
        // or format: '${value:,.0f}' for currency
    }
}
```

### Tooltip (Standard)
```javascript
tooltip: {
    shared: false,  // true for multi-series
    useHTML: true,
    headerFormat: '<b>{point.x:%b %e, %Y}</b><br/>',  // Date header
    pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y:,.2f}</b>',
    // For stacked charts, add footerFormat:
    footerFormat: '<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ccc;"><b>Total: {point.total:,.0f}</b></div>'
}
```

**Note:** Avoid custom `formatter` functions when possible. Ensure data is pre-formatted correctly (e.g., negative values if needed) rather than transforming in tooltip formatters.

### Legend
```javascript
legend: {
    enabled: true,  // false to hide
    align: 'center',
    verticalAlign: 'top',  // or 'bottom'
    layout: 'horizontal',
    itemStyle: {
        fontSize: '1rem'
    }
}
```

### Credits
```javascript
credits: {
    enabled: true,
    text: 'Data: Flipside',
    style: {
        fontSize: '0.75rem'
    }
}
```

## Chart Type Configurations

**Module Requirements:** Most chart types only need `highcharts.js`. Special types require additional modules:
- **Heatmap:** `heatmap.js`, `data.js`, `boost.js`
- **Bubble:** `highcharts-more.js` (required)
- **Sankey:** `sankey.js`
- **Export features:** `exporting.js`, `export-data.js` (if needed)

### Column/Bar Chart
**Modules:** `highcharts.js` only

```javascript
chart: {
    type: 'column',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
plotOptions: {
    column: {
        borderWidth: 0,
        pointPadding: 0.1,
        groupPadding: 0.1,
        stacking: 'normal'  // For stacked bars
    }
}
```

### Grouped Column Chart (Side-by-Side Bars)
**Modules:** `highcharts.js` only

**Use Case:** Comparing multiple series across categories (e.g., different action types per platform)

**Key UX Improvements for Hover/Tooltip:**
- Tooltip triggers on x-axis hover (entire category area), not just when mouse matches bar y-position
- Small bars remain hoverable even when values are tiny relative to axis scale
- Tooltip shows all series values for the hovered category

```javascript
chart: {
    type: 'column',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
tooltip: {
    shared: true,  // Critical: triggers on x-axis hover, shows all series in group
    useHTML: true,
    followPointer: true,  // Tooltip follows mouse for easier interaction
    hideDelay: 0,  // Show immediately
    headerFormat: '<div style="text-align: center; margin-bottom: 8px; font-weight: bold; font-size: 14px;">{point.key}</div>',
    pointFormat: '<div style="text-align: left;"><span style="color:{series.color}">\u25CF</span> {series.name}: <b>${point.y:,.0f}</b></div>'
},
plotOptions: {
    column: {
        borderWidth: 0,
        pointPadding: 0.1,
        groupPadding: 0.15,
        minPointLength: 1,  // Minimum height in pixels for small bars to ensure hoverability
        stickyTracking: true,  // Makes tooltips easier to trigger
        stacking: null  // No stacking - columns grouped side by side
    }
},
xAxis: {
    categories: ['Category1', 'Category2', ...],  // Categories for grouping
    title: { text: 'Category Label', style: { fontSize: '1.125rem' } },
    labels: { style: { fontSize: '1rem' }, rotation: -45 }
},
yAxis: {
    title: { text: 'Value Label', style: { fontSize: '1.125rem', fontWeight: 'bold' } },
    labels: { style: { fontSize: '1rem' }, format: '${value:,.0f}' }
},
series: [
    {
        name: 'Series 1',
        data: [value1, value2, ...]  // Values per category
    },
    {
        name: 'Series 2',
        data: [value1, value2, ...]
    }
    // ... more series
]
```

**Important Settings Explained:**
- **`shared: true`**: Tooltip triggers when hovering anywhere along the x-axis for a category, showing all series values at once
- **`minPointLength: 1`**: Ensures very small bars have minimum height for hover detection
- **`stickyTracking: true`**: Tooltips trigger more easily when hovering near points, not just directly on them
- **`followPointer: true`**: Tooltip follows mouse cursor for better visibility
- **`stacking: null`**: Explicitly disables stacking so bars appear side-by-side

### Area Chart (Non-Stacked)
```javascript
chart: {
    type: 'area',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
plotOptions: {
    area: {
        lineColor: '#666666',
        lineWidth: 2,
        marker: { enabled: false },
        fillOpacity: 0.3
    }
}
```

### Area Chart (Stacked)
```javascript
plotOptions: {
    area: {
        stacking: 'normal',
        lineColor: '#666666',
        lineWidth: 2,
        marker: { enabled: false },
        fillOpacity: 0.3
    }
}
```

### Line Chart
```javascript
chart: {
    type: 'line',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
plotOptions: {
    line: {
        marker: {
            enabled: true,
            symbol: 'circle',
            radius: 3
        }
    }
}
```

### Heatmap
**Modules:** `heatmap.js`, `data.js`, `boost.js` required

**Data Format:** `[[x, y, value], ...]` - where value determines color intensity. Highcharts handles this directly without CSV conversion.

```javascript
chart: {
    type: 'heatmap',
    backgroundColor: '#ffffff'
},
boost: {
    useGPUTranslations: true
},
colorAxis: {
    min: 0,
    minColor: '#FFFFFF',
    maxColor: '#2775ca'  // Use appropriate color
},
xAxis: {
    type: 'datetime',  // or 'category'
    title: { text: 'Date', style: { fontSize: '1rem' } },
    labels: { format: '{value:%b %e}', style: { fontSize: '1rem' } }
},
yAxis: {
    title: { text: 'Hour', style: { fontSize: '1rem' } },
    labels: { style: { fontSize: '1rem' }, format: '{value}:00' },
    min: 0,
    max: 23
},
series: [{
    name: 'Volume',
    boostThreshold: 100,
    borderWidth: 0.5,
    borderColor: '#f0f0f0',
    data: rawData.map(d => [
        new Date(d.DAY_).getTime(),  // x position (timestamp)
        d.HOUR_,                      // y position (0-23)
        d.TOTAL_GAS_USED             // color intensity
    ])
}]
```

### Donut/Pie Chart
**Modules:** `highcharts.js` only

```javascript
chart: {
    type: 'pie',
    backgroundColor: '#ffffff'
},
plotOptions: {
    pie: {
        innerSize: '50%',  // Makes it a donut
        dataLabels: {
            enabled: true,
            format: '{point.name}: {point.percentage:.1f}%'
        }
    }
}
```

### Combo Chart (Bar + Line with Dual Y-Axes)
**Modules:** `highcharts.js` only

**Key Pattern:** Use array of yAxis objects, assign each series to specific yAxis index

```javascript
chart: {
    type: 'column',  // Base type, but series can override
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
yAxis: [
    {
        // Left Y axis (index 0)
        title: {
            text: 'Left Axis Label',
            style: { fontSize: '1.125rem', color: '#333333' }
        },
        labels: {
            formatter: function() {
                return (this.value / 1000000).toLocaleString() + 'M';
            },
            style: { fontSize: '1rem', color: '#333333' }
        },
        plotLines: [{  // Optional: zero line
            value: 0,
            color: '#999999',
            width: 1,
            zIndex: 4
        }]
    },
    {
        // Right Y axis (index 1)
        title: {
            text: 'Right Axis Label',
            style: { fontSize: '1.125rem', color: '#2775ca' }
        },
        labels: {
            formatter: function() {
                return (this.value / 1000000).toLocaleString() + 'M';
            },
            style: { fontSize: '1rem', color: '#2775ca' }
        },
        opposite: true  // Required for right-side axis
    }
],
plotOptions: {
    column: {
        borderWidth: 0,
        groupPadding: 0.1,
        pointPadding: 0.05
    },
    line: {
        lineWidth: 3,
        marker: { enabled: false }
    }
},
series: [
    {
        name: 'Bar Series',
        type: 'column',  // Explicitly set type
        data: barData,
        yAxis: 0  // Use left axis
    },
    {
        name: 'Line Series',
        type: 'line',  // Different type
        data: lineData,
        yAxis: 1,  // Use right axis
        zIndex: 5  // Draw on top
    }
]
```

### Scatter Chart
**Modules:** `highcharts.js` only

**Data Format:** `[[x, y], ...]` - pairs of numeric values

```javascript
chart: {
    type: 'scatter',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
xAxis: {
    title: { text: 'X Axis Label', style: { fontSize: '1.125rem' } },
    labels: {
        style: { fontSize: '1rem' },
        formatter: function() {
            return this.value.toLocaleString();
        }
    },
    gridLineWidth: 1
},
yAxis: {
    title: { text: 'Y Axis Label', style: { fontSize: '1.125rem' } },
    labels: {
        style: { fontSize: '1rem' },
        formatter: function() {
            return this.value.toLocaleString();
        }
    }
},
plotOptions: {
    scatter: {
        marker: {
            radius: 4,
            states: {
                hover: {
                    enabled: true,
                    lineColor: 'rgb(100,100,100)'
                }
            }
        },
        states: {
            hover: {
                marker: { enabled: false }
            }
        },
        tooltip: {
            headerFormat: '<b>Point Details</b><br/>',
            pointFormat: 'X: <b>{point.x:,.0f}</b><br/>Y: <b>{point.y:,.0f}</b>'
        }
    }
},
series: [{
    name: 'Data Points',
    data: rawData.map(d => [d.X_VALUE, d.Y_VALUE]),
    color: '#2775ca'
}]
```

### Bubble Chart
**Modules:** `highcharts.js`, `highcharts-more.js` required

**Data Format:** `[[x, y, z], ...]` - where z is bubble size

```javascript
chart: {
    type: 'bubble',
    zoomType: 'xy',
    panning: { enabled: true, type: 'xy' },
    panKey: 'shift',
    backgroundColor: '#ffffff'
},
xAxis: {
    type: 'datetime',  // or numeric
    title: { text: 'X Axis Label', style: { fontSize: '1.125rem' } },
    labels: { format: '{value:%b %e}', style: { fontSize: '1rem' } }
},
yAxis: {
    title: { text: 'Y Axis Label', style: { fontSize: '1.125rem' } },
    labels: {
        style: { fontSize: '1rem' },
        formatter: function() {
            return this.value.toLocaleString();
        }
    }
},
tooltip: {
    useHTML: true,
    headerFormat: '<b>{series.name}</b><br/>',
    pointFormat: 'X: {point.x:%b %e, %Y}<br/>Y: <b>{point.y:,.2f}</b><br/>Size: <b>{point.z:,.2f}</b>'
},
plotOptions: {
    bubble: {
        minSize: 5,
        maxSize: 50
    }
},
series: [{
    name: 'Series Name',
    data: rawData.map(d => [
        new Date(d.DATE).getTime(),  // x
        d.Y_VALUE,                   // y
        d.SIZE_VALUE                 // z (bubble size)
    ])
}]
```

### Sankey Diagram
**Modules:** `sankey.js` required

**Data Format:** Nodes array + links array `[from, to, weight]`

```javascript
chart: {
    type: 'sankey',
    backgroundColor: '#ffffff',
    height: 750  // Sankey charts often need explicit height
},
tooltip: {
    useHTML: true,
    headerFormat: '<div style="text-align: center;"><b>{point.fromNode.name} → {point.toNode.name}</b></div>',
    pointFormat: '<div style="text-align: right;"><b>${point.weight:,.0f}</b></div>',
    style: { fontSize: '14px' }
},
series: [{
    keys: ['from', 'to', 'weight'],
    data: rawData.map(d => [d.SOURCE, d.DESTINATION, d.VALUE]),
    nodes: [
        { id: 'node1', name: 'Node 1', color: '#2775ca' },
        { id: 'node2', name: 'Node 2', color: '#f7941d' }
        // ... more nodes
    ],
    nodeWidth: 25,
    nodePadding: 15,
    minLinkWidth: 3,
    linkOpacity: 0.5,
    states: {
        hover: {
            linkOpacity: 0.7
        }
    }
}]
```

## Data Formatting Conventions

### Number Formatting
- **Tooltips:** 2 decimal places (`{point.y:,.2f}`)
- **Axis labels:** Whole numbers with commas (`{value:,.0f}`)
- **Currency:** Prefix with `$` (`${value:,.0f}`)

### Date Formatting
- **X-axis labels:** `{value:%b %e}` → "Oct 1"
- **Tooltip headers:** `{point.x:%b %e, %Y}` → "Oct 1, 2025"
- **Weekly labels:** `{value:%b %Y}` → "Oct 2025"

### Data Transformation Patterns

**Time Series (Single Series):**
```javascript
const series = [{
    name: 'Series Name',
    data: rawData.map(d => [
        new Date(d.DATE).getTime(),
        d.VALUE
    ])
}];
```

**Time Series (Multiple Series):**
```javascript
// Group by series identifier
const seriesData = {};
rawData.forEach(row => {
    const key = row.SERIES_KEY;
    if (!seriesData[key]) seriesData[key] = [];
    seriesData[key].push([
        new Date(row.DATE).getTime(),
        row.VALUE
    ]);
});

const series = Object.keys(seriesData).map(key => ({
    name: key,
    data: seriesData[key]
}));
```

**Stacked Bar/Category Data:**
```javascript
const categories = rawData.map(d => d.CATEGORY);
const series = [
    {
        name: 'Series 1',
        data: rawData.map(d => d.VALUE1)
    },
    {
        name: 'Series 2',
        data: rawData.map(d => d.VALUE2)
    }
];
```

**Combo Chart Data (Bar + Line):**
```javascript
// Separate data arrays for each series type
const barData = rawData.map(d => [
    new Date(d.DATE).getTime(),
    d.BAR_VALUE
]);
const lineData = rawData.map(d => [
    new Date(d.DATE).getTime(),
    d.LINE_VALUE
]);
```

**Scatter Chart Data:**
```javascript
const scatterData = rawData.map(d => [
    d.X_VALUE,  // Numeric x value
    d.Y_VALUE   // Numeric y value
]);
```

**Bubble Chart Data:**
```javascript
const bubbleData = rawData.map(d => [
    new Date(d.DATE).getTime(),  // x
    d.Y_VALUE,                   // y
    d.SIZE_VALUE                 // z (bubble size)
]);
```

**Sankey Data:**
```javascript
// Create nodes array with unique identifiers
const nodeSet = new Set();
rawData.forEach(d => {
    nodeSet.add(d.SOURCE);
    nodeSet.add(d.DESTINATION);
});

const nodes = Array.from(nodeSet).map(id => ({
    id: id,
    name: id.charAt(0).toUpperCase() + id.slice(1),  // Capitalize
    color: colorMap[id] || '#cccccc'
}));

// Links format: [from, to, weight]
const links = rawData.map(d => [d.SOURCE, d.DESTINATION, d.VALUE]);
```

## Interactive Features

### Zoom and Pan
Always enable for time series and large datasets:
```javascript
chart: {
    zoomType: 'xy',
    panning: {
        enabled: true,
        type: 'xy'
    },
    panKey: 'shift'  // Hold Shift to pan after zooming
}
```

**User Instructions:**
- Click and drag to zoom
- Hold `Shift` and drag to pan after zooming
- Use "Reset zoom" button to return to original view

## Export Configuration (PNG Export)

When user requests PNG export, use fixed sizing and transparent background:

```javascript
chart: {
    width: 1200,
    height: 800,
    backgroundColor: 'rgba(0,0,0,0)'  // Transparent
},
title: {
    style: {
        fontSize: '2.25rem',  // Equivalent to 36px at 16px base
        fontWeight: 'bold'
    }
},
// For export, rem units work fine (browser renders to pixels)
```

## Complete Example: Embedded Area Chart

```html
<div id="chart-container-volume">
    <div id="chart-volume"></div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    const rawData = [
        {"DAY_":"2025-10-01T00:00:00.000Z","ASSET":"USDC","VOLUME":1234567.89},
        {"DAY_":"2025-10-02T00:00:00.000Z","ASSET":"USDC","VOLUME":2345678.90}
    ];
    
    const seriesData = {};
    rawData.forEach(row => {
        const asset = row.ASSET;
        if (!seriesData[asset]) seriesData[asset] = [];
        seriesData[asset].push([
            new Date(row.DAY_).getTime(),
            row.VOLUME
        ]);
    });
    
    const series = Object.keys(seriesData).map(asset => ({
        name: asset,
        data: seriesData[asset]
    }));
    
    Highcharts.chart('chart-volume', {
        chart: {
            type: 'area',
            zoomType: 'xy',
            panning: { enabled: true, type: 'xy' },
            panKey: 'shift',
            backgroundColor: '#ffffff'
        },
        colors: ['#2775ca', '#f7941d'],
        title: {
            text: 'Daily Transfer Volume by Asset',
            style: {
                fontSize: '1.5rem',
                fontWeight: 'bold',
                color: '#333333'
            }
        },
        subtitle: {
            text: 'USD transfer volumes over time',
            style: {
                fontSize: '1rem',
                color: '#666666'
            }
        },
        xAxis: {
            type: 'datetime',
            title: { text: 'Date', style: { fontSize: '1.125rem' } },
            labels: { format: '{value:%b %e}', style: { fontSize: '1rem' } }
        },
        yAxis: {
            title: {
                text: 'Volume (USD)',
                style: { fontSize: '1.125rem', fontWeight: 'bold' }
            },
            labels: { format: '${value:,.0f}', style: { fontSize: '1rem' } }
        },
        tooltip: {
            shared: false,
            useHTML: true,
            headerFormat: '<b>{point.x:%b %e, %Y}</b><br/>',
            pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>${point.y:,.2f}</b>'
        },
        legend: {
            layout: 'horizontal',
            align: 'center',
            verticalAlign: 'bottom'
        },
        plotOptions: {
            area: {
                lineColor: '#666666',
                lineWidth: 2,
                marker: { enabled: false },
                fillOpacity: 0.3
            }
        },
        series: series,
        credits: {
            enabled: true,
            text: 'Data: Flipside',
            style: { fontSize: '0.75rem' }
        }
    });
});
</script>
```

## Data Injection Workflow

When SQL query results need to be injected into HTML templates (to avoid slow agent data duplication), use a placeholder-based workflow:

### Template Format

Create HTML templates with a placeholder where data should be injected:

```html
<script>
    document.addEventListener('DOMContentLoaded', function () {
        // Placeholder: {{JSON_DATA}}
        const rawData = {{JSON_DATA}};
        
        // Data transformation
        const chartData = rawData.results.map(item => ({
            x: new Date(item.DATE).getTime(),
            y: item.VALUE
        }));
        
        // Highcharts configuration
        Highcharts.chart('chart-id', {
            // ... config
        });
    });
</script>
```

### Placeholder Convention

- **Default placeholder:** `{{JSON_DATA}}`
- **Alternative placeholders:** Can be customized (e.g., `{{DATA}}`, `{{QUERY_RESULTS}}`)
- **Format:** Placeholder will be replaced with: `const jsonData = { ... };`

### Injection Script

Use `json_inject.py` (or equivalent) to inject data:

```bash
python json_inject.py template.html data.json output.html
python json_inject.py template.html data.json output.html --placeholder "{{DATA}}"
```

### SQL Query Result Format

SQL queries return JSON in this format:
```json
{
  "status": "completed",
  "results": [
    {"COLUMN1": "value1", "COLUMN2": 123},
    {"COLUMN1": "value2", "COLUMN2": 456}
  ],
  "queryId": "...",
  "aggregates": { ... }
}
```

Access data via `rawData.results` array in your JavaScript transformation code.

### Workflow Steps

1. **Agent generates HTML template** with `{{JSON_DATA}}` placeholder
2. **SQL query executes** via MCP tool (returns JSON response)
3. **Save JSON response** to file (e.g., `query_results.json`)
4. **Python script injects** JSON data into template: `python json_inject.py template.html query_results.json output.html`
5. **Final HTML file** contains embedded data ready for rendering

**Note:** SQL query results are returned as JSON objects with `status`, `results`, `queryId`, and `aggregates` fields. Access data via `rawData.results` in your JavaScript code.

This approach minimizes agent token usage and speeds up visualization generation.

## Best Practices Checklist

- [ ] Use unique chart container and chart IDs (e.g., `chart-container-{id}`, `chart-{id}`)
- [ ] Embed data as JavaScript constants (no external API calls)
- [ ] Enable zoom/pan for time series charts (`zoomType: 'xy'`, `panning: { enabled: true }`)
- [ ] Format numbers with commas and appropriate decimal places
- [ ] Use contextual colors from palette when applicable
- [ ] Include "Data: Flipside" credit
- [ ] Set appropriate font sizes (1.5rem title, 1rem subtitle, 1rem labels) - use rem units
- [ ] Use `useHTML: true` for tooltips with custom formatting
- [ ] For stacked charts, include total in tooltip footer (`footerFormat`)
- [ ] Set `backgroundColor: '#ffffff'` for white background
- [ ] Use `shared: true` tooltip for multi-series charts
- [ ] For grouped column charts, use `shared: true`, `minPointLength: 1`, and `stickyTracking: true` for better hover UX
- [ ] Rotate x-axis labels (-45°) if crowded
- [ ] For combo charts, assign each series to correct yAxis index (`yAxis: 0` or `yAxis: 1`)
- [ ] For Sankey charts, set explicit height (typically 600-800px)
- [ ] For heatmaps, ensure required modules are loaded (`heatmap.js`, `data.js`, `boost.js`) and use direct array format `[[x, y, value], ...]`
- [ ] For bubble charts, include `highcharts-more.js` module
- [ ] Avoid custom tooltip formatters - pre-format data correctly instead

## Chart Type Selection Guide

**When to use each type:**

- **Column/Bar:** Comparing discrete categories or time periods
- **Grouped Column:** Comparing multiple series across categories side-by-side (e.g., different action types per platform)
- **Stacked Bar:** Showing composition across categories
- **Area:** Emphasizing volume/magnitude over time (non-stacked) or cumulative totals (stacked)
- **Line:** Showing trends, especially with many data points
- **Combo (Bar+Line):** When you need to show two metrics with different scales (e.g., volume + cumulative)
- **Scatter:** Exploring relationships between two numeric variables
- **Bubble:** Adding a third dimension (size) to scatter plot
- **Heatmap:** Showing patterns across two dimensions (e.g., day × hour)
- **Donut/Pie:** Showing proportions of a whole (use sparingly, max 5-7 categories)
- **Sankey:** Showing flows between nodes (e.g., bridge flows, user journeys)

## Common Pitfalls and Solutions

**Problem:** Chart doesn't render
- **Solution:** Ensure Highcharts library is loaded in parent document, check console for errors
- **Solution:** Verify unique IDs don't conflict with other charts

**Problem:** Data not displaying correctly
- **Solution:** Check data format matches chart type requirements (`[[x,y]]` vs `[y]` vs `[[x,y,z]]`)
- **Solution:** For datetime axes, ensure dates are converted to timestamps: `new Date(d.DATE).getTime()`

**Problem:** Dual y-axes not working
- **Solution:** Use array format for `yAxis`, set `opposite: true` for right axis
- **Solution:** Explicitly assign each series: `yAxis: 0` or `yAxis: 1`

**Problem:** Tooltip formatting issues
- **Solution:** Use `useHTML: true` for custom HTML formatting
- **Solution:** For shared tooltips, use `shared: true` and `formatter` function

**Problem:** Colors not matching palette
- **Solution:** Explicitly set `colors` array in chart config
- **Solution:** For series-specific colors, set `color` property in series object

**Problem:** Heatmap performance issues
- **Solution:** Ensure `boost.js` module is loaded
- **Solution:** Consider data aggregation for very large datasets

