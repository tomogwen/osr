// Make an instance of two and place it on the page.
var elem = document.getElementById('draw-shapes');
var params = { width: 560, height: 560 };
var two = new Two(params).appendTo(elem);

var http = new XMLHttpRequest();
var url = "http://tomogwen.me:1234";

http.open("POST", url, true);

var rect = two.makeRectangle(280, 280, 560, 560);

http.onreadystatechange = function() {//Call a function when the state changes.
    if(http.readyState == 4 && http.status == 200) {
        alert(http.responseText);
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

rect.fill = 'rgb(210, 210, 210)';
rect.opacity = 0.75;
rect.noStroke();
two.makeLine(280,10,  280,550);
two.makeLine(10, 280, 550,280);

document.addEventListener("mousemove", function(event){
    event.preventDefault();
    if (event.which == 1) {
      var circle = two.makeCircle(event.pageX-10, event.pageY-10, 1);
      if ( (Math.floor((event.pageY-10)/20) < 28 ) && (Math.floor((event.pageX-10)/20) < 28 ) )  {
        mapArray[Math.floor((event.pageX-10)/20)][Math.floor((event.pageY-10)/20)] = 0;
      }
    }
});

document.addEventListener("click", function(event){
    event.preventDefault();
    two.clear();
    var rect = two.makeRectangle(280, 280, 560, 560);
    rect.fill = 'rgb(210, 210, 210)';
    rect.opacity = 0.75;
    rect.noStroke();
    two.makeLine(280,10,  280,550);
    two.makeLine(10, 280, 550,280);

    mapFlat = flatten(mapArray);
    console.log(mapFlat);

    http.open("POST", url, true);
    http.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
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
