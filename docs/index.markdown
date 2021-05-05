---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Admissions Analysis
heading: Law School Data Admissions Analyses (2017/2018 - 2020/2021)
---

<div style="margin: 35px 100px 18px 100px; font-family: calibri; text-align: justify">
    
  The plots below pull self-reported data from <a href="https://lawschooldata.org" target="_blank">LawSchoolData.org</a> and utilize 
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

<div style="margin: 0px 100px 18px 100px; font-family: calibri; text-align: justify">

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
<div style="margin: 0px 100px 18px 100px; font-family: calibri; text-align: justify">

  <i>Note</i>: Wait times are calculated for each group (<i>All</i>, <i>Accepted</i>, <i>Rejected</i>, <i>Waitlisted</i>) 
    by averaging the number of days from the time an applicant sent their application to the time the applicant receieved a decision. See plot below for a further breakdown of wait time data.

</div>
___

{% include waitbar.html %}
<div style="margin: 0px 100px 18px 100px; font-family: calibri; text-align: justify">

  <i>Note</i>: Error bars represent one standard deviation. Use this plot in conjunction with the plot above.

</div>
___

{% include splitters.html %}
<div style="margin: 0px 100px 18px 100px; font-family: calibri; text-align: justify">

  <i>Note</i>: The number of splitters and reverse splitters who applied and were admitted is for several
    schools so small as to make meaningful inferences impossible. For each school, <i>Index</i> values are calculated 
    by dividing the acceptance percentage of splitters or reverse splitters by the acceptance percentage of applicants
    who were neither. The greater above 1.0 a value is the <i>easier</i> it was for splitters or reverse splitters to 
    gain admission, compared to regular applicants; the lower below 1.0 a value is the <i>harder</i> it was. Splitters 
    are applicants with an LSAT score greather than the 75th percentile and a GPA less than the 25th percentile; 
    reverse splitters have a low LSAT and high GPA. Table columns may be rearranged by dragging.

</div>
___

{% include poolscatter.html %}
<div style="margin: 0px 100px 18px 100px; font-family: calibri; text-align: justify">

  <i>Note</i>: Percentages calculated by dividing the total number of applicants and admits, as determined by the 
    2020 acceptance rate and yield as reported by the ABA and made available in convenient format by 
    <a href="https://7sage.com/top-law-school-admissions/" target="_blank">7Sage</a>. An 
    applicant counts as having used LawSchoolData.org to report their applying to or hearing from a given school 
    if they entered at minimum the date they sent their application, their LSAT score, and their GPA.

</div>
