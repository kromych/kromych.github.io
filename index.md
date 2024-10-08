---
layout: default
---

<style type="text/css">
    .post-excerpt {
        background-color: #1c1c1c;
        color: #00ff00;
        font-family: "Courier New", Courier, monospace;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #00ff00;
        box-shadow: 0px 0px 10px rgba(0, 255, 0, 0.3);
        transition: background-color 0.3s, color 0.3s;
        display: inline-block;
        width: fit-content;
        max-width: 100%;
    }

    .post-excerpt:hover {
        background-color: #333333;
        color: #33ff33;
        box-shadow: 0px 0px 15px rgba(0, 255, 0, 0.5);
    }

    .post-excerpt p {
        margin: 0;
        line-height: 1.6;
    }
</style>

<h2>Posts</h2>
<ul>
    <h3>{{ page.category }}</h3>

    {% for post in site.posts %}
        <h3>{{ post.date | date_to_string }} &nbsp; <a href="{{ post.url }}">{{ post.title }}</a></h3>
        <div class="post-excerpt">
            {{ post.summary }}
        </div>
    {% endfor %}
</ul>

