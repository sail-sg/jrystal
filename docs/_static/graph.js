document.addEventListener("DOMContentLoaded", function() {
  fetch("/_static/graph.dot?" + new Date().getTime())
    .then(function(response) {
      if (!response.ok) {
        throw new Error("Network response was not ok: " + response.statusText);
      }
      return response.text();
    })
    .then(function(dot) {
      d3.select("#graph").graphviz().renderDot(dot)
        .on("end", function() {
          // Attach click event to every node.
          d3.selectAll("g.node").on("click", function(event) {
            var clickedLabel = d3.select(this).select("title").text().trim();

            // Toggle: if this node is already active, clear all styling and remove doc links.
            if (d3.select(this).classed("active")) {
              d3.selectAll("g.node").classed("active", false).classed("dim", false);
              d3.selectAll("g.edge").classed("active", false).classed("dim", false);
              d3.selectAll("svg a.doc-link").remove();
              return;
            }

            // Clear previous state.
            d3.selectAll("g.node").classed("active", false).classed("dim", false);
            d3.selectAll("g.edge").classed("active", false).classed("dim", false);
            d3.selectAll("svg a.doc-link").remove();

            // Mark the clicked node as active.
            d3.select(this).classed("active", true);

            // Dim every other node.
            d3.selectAll("g.node").filter(function() {
              var nodeLabel = d3.select(this).select("title").text().trim();
              return nodeLabel !== clickedLabel;
            }).classed("dim", true);

            // Dim edges as needed.
            d3.selectAll("g.edge").classed("dim", true);

            // Construct the URL from the node label.
            var baseUrl = "http://localhost:8000/index.html#";
            var url = baseUrl + encodeURIComponent(clickedLabel);

            // Compute the absolute position of the node's right edge.
            // First, get the bounding box of the clicked node.
            var bbox = this.getBBox();
            // Create an SVGPoint for the desired position: 5px to the right of the node, vertically centered.
            var pt = this.ownerSVGElement.createSVGPoint();
            pt.x = bbox.x + bbox.width + 0.1;
            pt.y = bbox.y + bbox.height / 2;
            // Transform the point using the node's current transformation matrix.
            var globalPt = pt.matrixTransform(this.getCTM());

            // Append the doc link to the top-level SVG container.
            d3.select(this.ownerSVGElement)
              .append("svg:a")
              .attr("class", "doc-link")
              .attr("xlink:href", url)
              .attr("target", "_blank")
              .append("svg:text")
              .text("doc")
              .attr("x", globalPt.x)
              .attr("y", globalPt.y);
          });
        });
    })
    .catch(function(error) {
      console.error("Error fetching DOT file:", error);
    });
});
