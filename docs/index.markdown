---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Admissions Analysis
heading: Law School Data Admissions Analyses (2017/2018 - 2020/2021)
---
<div style="margin: 35px 100px 18px 100px; font-family:calibri">
The graphs below pull self-reported data from <a href="https://lawschooldata.org" target="_blank">LawSchoolData.org</a> and utilize 
<a href="https://plotly.com" target="_blank">Plotly</a> to visualize different slices of this data. Plotly makes possible an 
interactive experience, among other things allowing users to: pan and zoom, using the toolbar at the top of each plot; 
adjust the x and y axes, by hovering over an axis at the middle or at either end; and highlight and hide the data sets 
represented in the legend, by single- or double-clicking on the traces.

<p></p>

Those interested in examining the source code may wish to visit the 
<a href="https://github.com/PlatosTwin/LawSchoolData/tree/gh-pages" target="_blank">GitHub repository</a> for this project. Feedback, 
suggestions, and other notes may be directed to 
<a href="https://www.reddit.com/user/IneffablePhilospoher" target="_blank">u/IneffablePhilosopher</a>.

<p></p>

See the footer for attributions and the latest date of update.
</div>
___
{% include timeline.html %}

<div style="margin: 0px 100px 18px 100px; font-family:calibri">
<i>Note</i>: Historical percentages and likelihood are calculated based on the past three cycles (17/18, 18/19, and 
19/20). The <i>Notified</i> trace includes only those who have received an acceptance, rejection, or waitlist—thus 
does not include withdrawls or holds. The <i>A</i>, <i>R</i>, and <i>WL</i> traces are calculated by taking the number of 
acceptances, rejections, or waitlists up to a given date and dividing by the total number. <i>Acceptance Likelihood</i> 
is calcuated by dividing the number of acceptances remaining at any given point by the number of applicants who at that point 
had not yet received an acceptance, rejection, or waitlist and who had not withdrawn. The faint diagonal lines 
against the background represent months of waiting: y=x (0 months), 1 month, 2 months, ..., 6 months. Marker outlines in
<span style="color: blue"><i>blue</i></span> represent splitters (>75th percentile LSAT and <25th percentile GPA) while marker 
outlines in <i>black</i> represent reverse splitters (<25th percentile LSAT and >75th percentile GPA).
</div>
___
{% include waithistogram.html %}
___
{% include waitbar.html %}
___
{% include splitters.html %}
___
{% include poolscatter.html %}

