// Make an instance of two and place it on the page.
var elem = document.getElementById('draw-shapes');
var params = { width: 280, height: 280 };
var two = new Two(params).appendTo(elem);

// two has convenience methods to create shapes.

var rect = two.makeRectangle(140, 140, 280, 280);

// The object returned has many stylable properties:


rect.fill = 'rgb(210, 210, 210)';
rect.opacity = 0.75;
rect.noStroke();
two.makeLine(140,5,   140,275);
two.makeLine(5,  140, 275,140);


document.addEventListener("mousemove", function(event){
    event.preventDefault();
    if (event.which == 1) {
      var circle = two.makeCircle(event.pageX-10, event.pageY-10, 1);
    }

});

document.addEventListener("click", function(event){
    event.preventDefault();
    two.clear();
    var rect = two.makeRectangle(140, 140, 280, 280);
    rect.fill = 'rgb(210, 210, 210)';
    rect.opacity = 0.75;
    rect.noStroke();
    two.makeLine(140,5,   140,275);
    two.makeLine(5,  140, 275,140);
});

// Don't forget to tell two to render everything
// to the screen

two.bind('update', function(frameCount) {

}).play();  // Finally, start the animation loop
two.update();
