import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';
import { api } from '../services/api';
import { AppData, Priority } from '../types';

vi.mock('../services/api');

const mockData: AppData = {
  user: { id: 1, full_name: 'Test User', email: 'test@example.com' },
  areas: [{ id: 1, name: 'Work', color: 'bg-red-500' }],
  tasks: [{ id: 1, title: 'Test Task', area_id: 1, completed: false, description: '', due_date: '', priority: Priority.High }],
  notes: [{ id: 1, title: 'Test Note', area_id: 1, content: '' }],
};

describe('App Component', () => {
  let originalDate: DateConstructor;

  beforeEach(() => {
    vi.resetAllMocks();
    localStorage.clear();

    // Mock Date to return a fixed morning time (April 17, 2026, 10:00 AM)
    originalDate = global.Date;
    const mockDate = new originalDate(2026, 3, 17, 10);
    global.Date = class extends originalDate {
      constructor(...args: any[]) {
        if (args.length === 0) return mockDate;
        return new originalDate(...args);
      }
    } as DateConstructor;
  });

  afterEach(() => {
    global.Date = originalDate;
  });

  it('shows Login page when not authenticated', () => {
    render(<App />);
    expect(screen.getByText('Welcome back')).toBeInTheDocument();
  });

  it('switches to SignUp page', async () => {
    render(<App />);
    await userEvent.click(screen.getByText('Sign up'));
    expect(screen.getByText('Create an account')).toBeInTheDocument();
  });

  it('fetches data and renders dashboard on successful login', async () => {
    localStorage.setItem('focusflow_token', 'test-token');
    (api.getUser as vi.Mock).mockResolvedValue(mockData.user);
    (api.getAreas as vi.Mock).mockResolvedValue(mockData.areas);
    (api.getTasks as vi.Mock).mockResolvedValue(mockData.tasks);
    (api.getNotes as vi.Mock).mockResolvedValue(mockData.notes);

    render(<App />);

    await waitFor(() => {
      const greeting = screen.getByText(/Good morning/, { selector: 'h1' });
      expect(greeting).toBeInTheDocument();
      expect(greeting).toHaveTextContent(/Test/);
    });
  });

  it('logs out and returns to login page', async () => {
    localStorage.setItem('focusflow_token', 'test-token');
    (api.getUser as vi.Mock).mockResolvedValue(mockData.user);
    (api.getAreas as vi.Mock).mockResolvedValue(mockData.areas);
    (api.getTasks as vi.Mock).mockResolvedValue(mockData.tasks);
    (api.getNotes as vi.Mock).mockResolvedValue(mockData.notes);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Test User'));
    await userEvent.click(screen.getByText('Log out'));

    expect(screen.getByText('Welcome back')).toBeInTheDocument();
    expect(localStorage.getItem('focusflow_token')).toBeNull();
  });

  it('opens and saves a new task', async () => {
    localStorage.setItem('focusflow_token', 'test-token');
    (api.getUser as vi.Mock).mockResolvedValue(mockData.user);
    (api.getAreas as vi.Mock).mockResolvedValue(mockData.areas);
    (api.getTasks as vi.Mock).mockResolvedValue(mockData.tasks);
    (api.getNotes as vi.Mock).mockResolvedValue(mockData.notes);
    (api.createTask as vi.Mock).mockResolvedValue({});

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('New Task')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('New Task'));
    expect(screen.getByPlaceholderText('Task title')).toBeInTheDocument();

    await userEvent.type(screen.getByPlaceholderText('Task title'), 'A brand new task');
    await userEvent.click(screen.getByText('Save Changes'));

    expect(api.createTask).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(screen.queryByPlaceholderText('Task title')).not.toBeInTheDocument();
    });
  });

  it('navigates between pages', async () => {
    localStorage.setItem('focusflow_token', 'test-token');
    (api.getUser as vi.Mock).mockResolvedValue(mockData.user);
    (api.getAreas as vi.Mock).mockResolvedValue(mockData.areas);
    (api.getTasks as vi.Mock).mockResolvedValue(mockData.tasks);
    (api.getNotes as vi.Mock).mockResolvedValue(mockData.notes);

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    // Navigate to Tasks
    await userEvent.click(screen.getByText('Tasks'));
    await waitFor(() => {
      expect(screen.getByText('My Tasks')).toBeInTheDocument();
    });

    // Navigate to Notes
    await userEvent.click(screen.getByText('Notes'));
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 2, name: 'Notes' })).toBeInTheDocument();
    });

    // Click the sidebar "Work" button (first of two "Work" buttons)
    const workButtons = screen.getAllByText('Work');
    await userEvent.click(workButtons[0]);

    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'Work' })).toBeInTheDocument();
    });
  });
});