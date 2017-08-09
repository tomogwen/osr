// Make an instance of two and place it on the page.
var elem = document.getElementById('draw-shapes');
var params = { width: 560, height: 560 };
var two = new Two(params).appendTo(elem);

var http = new XMLHttpRequest();
var url = "http://tomogwen.me:1234";
var areButtons = 0;


http.open("POST", url, true);

function textToLabel (text) {
    var number;
    switch(text) {
      case "Positive Linear":
        number = 0;
        break;
      case "Negative Linear":
        number = 1;
        break;
      case "Positive Quadratic":
        number = 2;
        break;
      case "Negative Quadratic":
        number = 3;
        break;
      case "Sine":
        number = 4;
        break;
      case "Cosine":
        number = 5;
        break;
      default:
        number = 9;
    }
    return number;
}

function checkResponse(mapArray, bestGuess, mapArraySaved) {
    document.getElementById("request").innerHTML = "Help me learn. Was I correct?";

    if (areButtons == 0) {
      var button = document.createElement("input");
      button.type = "button";
      button.value = "Yes";
      button.name = "correct";
      button.onclick = function() {

        mapFlat = flatten(mapArraySaved);
        mapFlat.unshift(1,textToLabel(bestGuess));
        http.open("POST", url, true);
        http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        mapFlat = mapFlat.toString();
        mapFlat = mapFlat.replace(/,/g , "");
        http.send(mapFlat);

        document.getElementById("buttons").removeChild(button)
        document.getElementById("buttons").removeChild(button1)
        areButtons = 0;

        for(var i = 0; i < 28; i++) {
          for(var j = 0; j < 28; j++) {
            mapArraySaved[i][j] = 1;
          }
        }
      }
      document.getElementById("buttons").appendChild(button)
      var button1 = document.createElement("input");
      button1.type = "button";
      button1.value = "No";
      button1.name = "correct";
      button1.onclick = function() {

        var newList = document.createElement("select");
        newList.name = "optionList";
        newList.id = "optionList";
        newList.appendChild(new Option("Positive Linear", "0"));
        newList.appendChild(new Option("Negative Linear", "1"));
        newList.appendChild(new Option("Positive Quadratic", "2"));
        newList.appendChild(new Option("Negative Quadratic", "3"));
        newList.appendChild(new Option("Sine", "4"));
        newList.appendChild(new Option("Cosine", "5"));

        document.getElementById("buttons").appendChild(newList);

        var button2 = document.createElement("input");
        button2.type = "button";
        button2.value = "Submit";
        button2.name = "submit";
        button2.onclick = function() {

          mapFlat = flatten(mapArraySaved);
          mapFlat.unshift(1, document.getElementById("optionList").value);
          http.open("POST", url, true);
          http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
          mapFlat = mapFlat.toString();
          mapFlat = mapFlat.replace(/,/g , "");
          http.send(mapFlat);

          document.getElementById("buttons").removeChild(button)
          document.getElementById("buttons").removeChild(button1)
          areButtons = 0;
          document.getElementById("buttons").removeChild(button2)
          document.getElementById("buttons").removeChild(newList)

          for(var i = 0; i < 28; i++) {
            for(var j = 0; j < 28; j++) {
              mapArraySaved[i][j] = 1;
            }
          }

        }
        document.getElementById("buttons").appendChild(button2)

      }
      document.getElementById("buttons").appendChild(button)
      document.getElementById("buttons").appendChild(button1)
      areButtons = 1;
    }

    for(var i = 0; i < 28; i++) {
      for(var j = 0; j < 28; j++) {
        mapArraySaved[i][j] = mapArray[i][j];
      }
    }
    // reset map array WARNING CHANGES RESET POSITION
    for(var i = 0; i < 28; i++) {
      for(var j = 0; j < 28; j++) {
        mapArray[i][j] = 1;
      }
    }
}


http.onreadystatechange = function() {//Call a function when the state changes.
    if(http.readyState == 4 && http.status == 200) {
        var bestGuess = http.responseText;
        console.log(bestGuess);

        if(bestGuess == "datainvalid") {
          document.getElementById("request").innerHTML = "Invalid data sent to API, please register an issue at http://github.com/tomogwen/osr";
        }
        else if (bestGuess == 'datasaved') {
          document.getElementById("request").innerHTML = "Thanks for making me better!";
        }
        else {
          document.getElementById("label").innerHTML = "I guessed: " + bestGuess;
          checkResponse(mapArray, bestGuess, mapArraySaved);
        }
    }
}

// The object returned has many stylable properties:

var mapArray = new Array();
for(var i = 0; i < 28; i++) {
  mapArray[i] = new Array;
  for(var j = 0; j < 28; j++) {
    mapArray[i][j] = 1;
  }
}
var mapArraySaved = new Array();
for(var i = 0; i < 28; i++) {
  mapArraySaved[i] = new Array;
  for(var j = 0; j < 28; j++) {
    mapArraySaved[i][j] = mapArray[i][j];
  }
}

function flatten(arr) {
  return arr.reduce(function (flat, toFlatten) {
    return flat.concat(Array.isArray(toFlatten) ? flatten(toFlatten) : toFlatten);
  }, []);
}


function drawAxis() {
  var rect = two.makeRectangle(280, 280, 560, 560);
  rect.fill = 'rgb(220, 220, 220)';
  rect.opacity = 0.75;
  rect.noStroke();
  var xax = two.makeLine(280,10,  280,550);
  var yax = two.makeLine(10, 280, 550,280); //*/

  /*var rect = two.makeRectangle(280, 280, 560, 560);
  rect.fill = 'rgb(159, 216, 245)';
  rect.opacity = 0.75;
  rect.noStroke();
  var xax = two.makeLine(280,10,  280,550);
  var yax = two.makeLine(10, 280, 550,280);
  xax.linewidth = 3;
  yax.linewidth = 3; //*/

}
drawAxis();

function drawCircle(y, x, position) {
  var circle = two.makeCircle(x, y, 1);
  circle.fill = "black"; //*/

  /*var circle = two.makeCircle(x, y, 3.5);
  circle.fill = "rgb(218, 74, 74)"; //*/
}


document.addEventListener("mousemove", function(event){
    event.preventDefault();
    if (event.which == 1) {

      var position = document.getElementById("draw-shapes").getBoundingClientRect();
      drawCircle(event.pageY-position.top, event.pageX-position.left, position);
      if ( (Math.floor((event.pageY-position.top)/20) < 28 ) && (Math.floor((event.pageX-position.left)/20) < 28 ) )  {
        mapArray[Math.floor((event.pageY-position.top)/20)][Math.floor((event.pageX-position.left)/20)] = 0;
      }
    }
});

document.addEventListener("click", function(event){
    event.preventDefault();
    if (event.pageX < 580 && event.pageY < 580) {
      two.clear();
      drawAxis();

      mapFlat = flatten(mapArray);
      mapFlat.unshift(0,0);
      http.open("POST", url, true);
      http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      mapFlat = mapFlat.toString();
      mapFlat = mapFlat.replace(/,/g , "");
      http.send(mapFlat);

    }
});


// Don't forget to tell two to render everything
// to the screen

two.bind('update', function(frameCount) {

}).play();  // Finally, start the animation loop
two.update();
