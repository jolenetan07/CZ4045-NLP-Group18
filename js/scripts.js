// SIDEBAR TOGGLE

var sidebarOpen = false;
var sidebar = document.getElementById("sidebar");

function openSidebar() {
  if(!sidebarOpen) {
    sidebar.classList.add("sidebar-responsive");
    sidebarOpen = true;
  }
}

function closeSidebar() {
  if(sidebarOpen) {
    sidebar.classList.remove("sidebar-responsive");
    sidebarOpen = false;
  }
}



// ---------- CHARTS ----------

// ----- Model 1 -----
// Model 1 Bar
var barChart1Options = {
  series: [{
    data: [400, 600]
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
    categories: ["Positive", "Negative"],
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
  series: [400, 600],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: ["#0000FF", "#FF0000"],
labels: ['Positive', 'Negative'],
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



// ----- Model 2 -----
// Model 2 Bar
var barChart2Options = {
  series: [{
    data: [400, 600]
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
    categories: ["Positive", "Negative"],
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
  series: [400, 600],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: ["#0000FF", "#FF0000"],
labels: ['Positive', 'Negative'],
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


// ----- Model 3 -----
// Model 3 Bar
var barChart3Options = {
  series: [{
    data: [400, 600]
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
    categories: ["Positive", "Negative"],
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
  series: [400, 600],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: ["#0000FF", "#FF0000"],
labels: ['Positive', 'Negative'],
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


// ----- Model 4 -----
// Model 4 Bar
var barChart4Options = {
  series: [{
    data: [400, 600]
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
    categories: ["Positive", "Negative"],
  },
  yaxis: {
    title: {
      text: "Count"
    }
  }
};

var barChart4 = new ApexCharts(document.querySelector("#bar-chart-4"), barChart4Options);
barChart4.render();


// Model 4 Pie
var pieChart4Options = {
  series: [400, 600],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: ["#0000FF", "#FF0000"],
labels: ['Positive', 'Negative'],
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

var pieChart4 = new ApexCharts(document.querySelector("#pie-chart-4"), pieChart4Options);
pieChart4.render();


// ----- Model 5 -----
// Model 5 Bar
var barChart5Options = {
  series: [{
    data: [400, 600]
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
    categories: ["Positive", "Negative"],
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
  series: [400, 600],
  chart: {
  width: 400,
  offsetX: 25,
  offsetY: 40,
  type: 'pie',
},
colors: ["#0000FF", "#FF0000"],
labels: ['Positive', 'Negative'],
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