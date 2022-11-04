// ---------- CHARTS ----------

// ----- vader -----
// Model 1 Bar
var barChart1Options = {
  series: [{
    data: [7652, 4163, 5816],
  }],
  chart: {
    type: 'bar',
    height: 350,
    toolbar: {
      show: false
    },
  },
  colors: [
    "#367952",
    "#f5b74f",
    "#cc3c43",
  ],
  plotOptions: {
    bar: {
      distributed: true,
      borderRadius: 4,
      horizontal: false,
      columnWidth: '40%',
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: ["Positive", "Neutral", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart1 = new ApexCharts(document.querySelector("#bar-chart-1"), barChart1Options);
barChart1.render();


// Model 1 Pie
var pieChart1Options = {
  series: [7652, 4163, 5816],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
// colors: ["#0000FF", "#FF0000"],
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
labels: ['Positive', 'Neutral', 'Negative'],
responsive: [{
  breakpoint: 480,
  options: {
    chart: {
      width: 200
    },
    legend: {
      position: 'bottom'
    }
  }
}]
};

var pieChart1 = new ApexCharts(document.querySelector("#pie-chart-1"), pieChart1Options);
pieChart1.render();



// ----- textblob -----
// Model 2 Bar
var barChart2Options = {
  series: [{
    data: [7888, 6028, 3715]
  }],
  chart: {
    type: 'bar',
    height: 350,
    toolbar: {
      show: false
    },
  },
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
  plotOptions: {
    bar: {
      distributed: true,
      borderRadius: 4,
      horizontal: false,
      columnWidth: '40%',
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: ["Positive", "Neutral", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart2 = new ApexCharts(document.querySelector("#bar-chart-2"), barChart2Options);
barChart2.render();


// Model 2 Pie
var pieChart2Options = {
  series: [7888, 6028, 3715],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
labels: ['Positive', 'Neutral', 'Negative'],
responsive: [{
  breakpoint: 480,
  options: {
    chart: {
      width: 200
    },
    legend: {
      position: 'bottom'
    }
  }
}]
};

var pieChart2 = new ApexCharts(document.querySelector("#pie-chart-2"), pieChart2Options);
pieChart2.render();


// ----- linearsvc -----
// Model 3 Bar
var barChart3Options = {
  series: [{
    data: [5088, 4042, 8501]
  }],
  chart: {
    type: 'bar',
    height: 350,
    toolbar: {
      show: false
    },
  },
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
  plotOptions: {
    bar: {
      distributed: true,
      borderRadius: 4,
      horizontal: false,
      columnWidth: '40%',
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: ["Positive", "Neutral", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart3 = new ApexCharts(document.querySelector("#bar-chart-3"), barChart3Options);
barChart3.render();


// Model 3 Pie
var pieChart3Options = {
  series: [5088, 4042, 8501],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
labels: ['Positive', 'Neutral', 'Negative'],
responsive: [{
  breakpoint: 480,
  options: {
    chart: {
      width: 200
    },
    legend: {
      position: 'bottom'
    }
  }
}]
};

var pieChart3 = new ApexCharts(document.querySelector("#pie-chart-3"), pieChart3Options);
pieChart3.render();


// ----- xgboost -----
// Model 5 Bar
var barChart5Options = {
  series: [{
    data: [5018, 3667, 8946]
  }],
  chart: {
    type: 'bar',
    height: 350,
    toolbar: {
      show: false
    },
  },
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
  plotOptions: {
    bar: {
      distributed: true,
      borderRadius: 4,
      horizontal: false,
      columnWidth: '40%',
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: ["Positive", "Neutral", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart5 = new ApexCharts(document.querySelector("#bar-chart-5"), barChart5Options);
barChart5.render();


// Model 5 Pie
var pieChart5Options = {
  series: [5018, 3667, 8946],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
labels: ['Positive', 'Neutral', 'Negative'],
responsive: [{
  breakpoint: 480,
  options: {
    chart: {
      width: 200
    },
    legend: {
      position: 'bottom'
    }
  }
}]
};

var pieChart5 = new ApexCharts(document.querySelector("#pie-chart-5"), pieChart5Options);
pieChart5.render();


// ----- complimentnb -----
// Model 6 Bar
var barChart6Options = {
  series: [{
    data: [5078, 4198, 8355]
  }],
  chart: {
    type: 'bar',
    height: 350,
    toolbar: {
      show: false
    },
  },
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
  plotOptions: {
    bar: {
      distributed: true,
      borderRadius: 4,
      horizontal: false,
      columnWidth: '40%',
    }
  },
  dataLabels: {
    enabled: false
  },
  legend: {
    show: false
  },
  xaxis: {
    categories: ["Positive", "Neutral", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart6 = new ApexCharts(document.querySelector("#bar-chart-6"), barChart6Options);
barChart6.render();


// Model 6 Pie
var pieChart6Options = {
  series: [5078, 4198, 8355],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: [
  "#367952",
  "#f5b74f",
  "#cc3c43",
],
labels: ['Positive', 'Neutral', 'Negative'],
responsive: [{
  breakpoint: 480,
  options: {
    chart: {
      width: 200
    },
    legend: {
      position: 'bottom'
    }
  }
}]
};

var pieChart6 = new ApexCharts(document.querySelector("#pie-chart-6"), pieChart6Options);
pieChart6.render();