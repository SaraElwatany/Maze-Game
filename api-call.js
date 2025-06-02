

function flattenLandmarks(landmarks) {

  const flat = [];

  for (const lm of landmarks) {
    flat.push(lm.x, lm.y, lm.z);
  }

  return flat;
}






async function getPredictedLabel(landmarks) {

  const processed_t = flattenLandmarks(landmarks);

  try {

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ landmarks: processed_t }),
    });

    const data = await response.json();
    let label = data.label;

    console.log("Predicted label:", label);

    if (label == "like"){
      label = "up";
    }
    else if (label == "dislike"){
      label = "down";
    }
    else if (label == "one"){
      label = "right";
    }
    else if (label == "peace"){
      label = "left";
    }
    else{
      label = null;
    }

    console.log("Modified label:", label);

    return label; // "up", "down", "left", "right", or null
  } 
  
  catch (error) {
    console.error("Error in prediction:", error);
    return null;
  }

}




// async function getPredictedLabel(processed_t) {
//   // TODO: Call your model's api here
//   // and return the predicted label
//   // Possible labels: "up", "down", "left", "right", null
//   // null means stop & wait for the next gesture
//   // For now, we will return a random label
//   const labels = ["up", "down", "left", "right"];
//   const randomIndex = Math.floor(Math.random() * labels.length);
//   const randomLabel = labels[randomIndex];
//   console.log("Predicted label:", randomLabel);
//   return randomLabel;
// }
