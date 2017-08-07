// Make an instance of two and place it on the page.
var elem = document.getElementById('draw-shapes');
var params = { width: 560, height: 560 };
var two = new Two(params).appendTo(elem);

var http = new XMLHttpRequest();
var url = "http://tomogwen.me:1234";

http.open("POST", url, true);


http.onreadystatechange = function() {//Call a function when the state changes.
    if(http.readyState == 4 && http.status == 200) {
        var bestGuess = http.responseText;
        console.log(bestGuess);
        document.getElementById("label").innerHTML = bestGuess;
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
    two.clear();
    drawAxis();

    mapFlat = flatten(mapArray);

    http.open("POST", url, true);
    http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    mapFlat = mapFlat.toString();
    mapFlat = mapFlat.replace(/,/g , "");
    http.send(mapFlat);

    for(var i = 0; i < 28; i++) {
      for(var j = 0; j < 28; j++) {
        mapArray[i][j] = 1;
      }
    }
});


// Don't forget to tell two to render everything
// to the screen

two.bind('update', function(frameCount) {

}).play();  // Finally, start the animation loop
two.update();
