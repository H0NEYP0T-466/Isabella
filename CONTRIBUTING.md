# Contributing to Isabella

Thank you for your interest in contributing to Isabella! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, screenshots, etc.)
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (OS, Node.js version, Python version, browser, etc.)
- **Include relevant logs** from the console or terminal

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) when creating bug reports.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful** to most users
- **List some examples** of how the enhancement would be used
- **Specify if this is something you'd be willing to implement**

Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml) when suggesting enhancements.

### Pull Requests

We actively welcome your pull requests! Here's how to submit them:

1. **Fork the repository** and create your branch from `main`
2. **Set up your development environment** (see [Development Setup](#development-setup))
3. **Make your changes** following our [Coding Guidelines](#coding-guidelines)
4. **Test your changes** thoroughly (see [Testing](#testing))
5. **Update documentation** if needed
6. **Commit your changes** with clear, descriptive commit messages
7. **Push to your fork** and submit a pull request

#### Pull Request Guidelines

- Follow the [pull request template](.github/pull_request_template.md)
- Link to relevant issues (e.g., "Fixes #123")
- Include screenshots or GIFs for UI changes
- Ensure all tests pass
- Update documentation as needed
- Keep PRs focused on a single concern
- Write clear commit messages following conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `style:` for formatting changes
  - `refactor:` for code refactoring
  - `test:` for adding tests
  - `chore:` for maintenance tasks

## Development Setup

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- MongoDB 7.0+
- Git

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run linter
npm run lint

# Build for production
npm run build
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with required variables
echo "LONGCAT_API_KEY=your_api_key_here" > .env

# Start the server
uvicorn main:app --reload --port 5000
```

### MongoDB Setup

```bash
# Using Docker (recommended)
docker run -d -p 27017:27017 --name mongodb mongo:7.0

# Or install MongoDB locally
```

## Coding Guidelines

### General Principles

- Write clean, readable, and maintainable code
- Follow the existing code style in the project
- Add comments for complex logic
- Keep functions small and focused
- Use meaningful variable and function names

### Frontend (TypeScript/React)

- Follow TypeScript best practices
- Use functional components with hooks
- Use proper TypeScript types (avoid `any`)
- Follow React best practices and hooks rules
- Run ESLint before committing: `npm run lint`
- Use CSS modules or styled-components for styling
- Keep components small and reusable

### Backend (Python/FastAPI)

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Write docstrings for classes and functions
- Use async/await for I/O operations
- Keep route handlers thin - move logic to service layer
- Use proper error handling with appropriate HTTP status codes
- Log important operations using the logger utility

### Code Formatting

- **Frontend**: ESLint configuration is provided in `eslint.config.js`
- **Backend**: Use Black or autopep8 for Python formatting
- Keep lines under 100 characters when possible
- Use 2 spaces for indentation in TypeScript/React
- Use 4 spaces for indentation in Python

## Testing

### Frontend Testing

```bash
# Run tests (when implemented)
npm test

# Run tests in watch mode
npm test -- --watch
```

### Backend Testing

```bash
# Navigate to backend directory
cd backend

# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_emotion_integration.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests

- Write tests for new features
- Update tests when changing existing features
- Aim for high test coverage on business logic
- Use meaningful test names that describe what is being tested
- Follow the existing test structure and patterns

### Manual Testing

Before submitting a PR:

1. Test the frontend UI thoroughly
2. Test all API endpoints with different inputs
3. Verify MongoDB operations work correctly
4. Test error handling and edge cases
5. Test on different browsers (if frontend changes)
6. Check console for errors or warnings

## Documentation

### Updating Documentation

- Update README.md for significant changes
- Update API documentation for new/changed endpoints
- Add/update comments in code for complex logic
- Update backend documentation in `backend/README.md` if needed
- Keep documentation in sync with code changes

### Writing Documentation

- Use clear and concise language
- Include code examples where appropriate
- Add screenshots or diagrams for complex features
- Keep formatting consistent with existing docs
- Proofread for grammar and spelling

## Community

### Getting Help

- Check existing [issues](https://github.com/H0NEYP0T-466/Isabella/issues)
- Review [documentation](README.md)
- Search for similar questions or problems
- Create a new issue if you need help

### Stay Updated

- Watch the repository for updates
- Review open issues and PRs
- Participate in discussions
- Follow project guidelines and best practices

## Review Process

All submissions require review before merging. We aim to:

- Review PRs within 1-2 days
- Provide constructive feedback
- Request changes if needed
- Merge approved PRs promptly

## Recognition

Contributors will be recognized in:

- Git commit history
- Release notes for significant contributions
- Project documentation (if applicable)

## Questions?

Feel free to reach out by:

- Opening an issue for questions
- Commenting on relevant issues or PRs
- Contacting project maintainers

---

Thank you for contributing to Isabella! 🎉
