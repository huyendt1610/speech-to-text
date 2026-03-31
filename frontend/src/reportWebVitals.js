// To collect Web Vitals indicators
const reportWebVitals = onPerfEntry => {
  if (onPerfEntry && onPerfEntry instanceof Function) {
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      getCLS(onPerfEntry); //Cumulative Layout Shift, how much layout shift when load
      getFID(onPerfEntry); // First Input Delay: how delay when users interact the first time 
      getFCP(onPerfEntry); // First Contentful Paint: when the first content is displayed
      getLCP(onPerfEntry); // Largest Contentful Paint: when the largest content is displayed
      getTTFB(onPerfEntry); //Time to First Byte response time from server
    });
  }
};

export default reportWebVitals;
