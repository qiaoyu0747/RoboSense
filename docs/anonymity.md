# Anonymous review preparation

This package uses neutral filenames and contains only text documentation. It includes no Git history, original source attachments, remote tracking assets, or external account links. ZIP entry timestamps are normalized to a fixed neutral date.

Before publication or after adding content:

- Replace the paper placeholder only with an anonymized review copy. Inspect PDF author, creator, subject, annotations, and embedded attachment metadata.
- Check text, figures, filenames, logs, notebook outputs, paths, and model metadata for identifying information.
- Remove identifying acknowledgments, contact details, personal or institutional links, and funding identifiers from review materials.
- Check repository owner information, commit history, branches, tags, issue templates, workflow badges, and external services. A clean archive cannot control hosting-platform metadata.
- Use a clean anonymous publication destination and review the rendered repository as a reviewer would see it.
- Confirm that dataset access routes and paper links preserve the intended review anonymity.

The `.gitignore` helps keep local artifacts out of future commits; it does not remove files that have already been committed or sanitize file contents.
