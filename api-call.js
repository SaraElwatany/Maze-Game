async function getPredictedLabel(processed_t) {
  // TODO: Call your model's api here
  // and return the predicted label
  // Possible labels: "up", "down", "left", "right", null
  // null means stop & wait for the next gesture
  // For now, we will return a random label
  const labels = ["up", "down", "left", "right"];
  const randomIndex = Math.floor(Math.random() * labels.length);
  const randomLabel = labels[randomIndex];
  console.log("Predicted label:", randomLabel);
  return randomLabel;
}
